import { useCallback, useEffect, useRef, useState } from "react";

/* ─── Types ──────────────────────────────────────────────────────────── */

interface JobStatus {
  job_id: string;
  video_id: string;
  status: "pending" | "processing" | "ready" | "failed";
  progress: number;
  message: string;
}

interface SearchHit {
  text: string;
  start_seconds: number;
  end_seconds: number;
  distance: number | null;
  snippet_index: number | null;
  chunk_index: number | null;
  source: "transcript" | "visual";
}

interface SearchResponse {
  query: string;
  video_id: string;
  hits: SearchHit[];
  best_window: { start_seconds: number; end_seconds: number; hit_count: number } | null;
  clip_url: string | null;
  visual_hits_count: number;
  transcript_hits_count: number;
}

/* ─── Helpers ────────────────────────────────────────────────────────── */

const API = "/api/v1";

function fmtTime(s: number): string {
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}`;
}

function MatchBadge({ distance }: { distance: number | null }) {
  if (distance === null) return null;
  const pct = Math.round(Math.max(0, 1 - distance) * 100);
  const bg = pct >= 70 ? "bg-emerald-500" : pct >= 40 ? "bg-amber-500" : "bg-slate-500";
  return (
    <span className={`text-[10px] font-bold px-2 py-0.5 rounded ${bg} text-white`}>
      {pct}%
    </span>
  );
}

function SourceBadge({ source }: { source: "transcript" | "visual" }) {
  if (source === "visual") {
    return (
      <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full bg-violet-500 text-white flex items-center gap-1">
        <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
          <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
          <circle cx="12" cy="12" r="3" />
        </svg>
        CLIP
      </span>
    );
  }
  return (
    <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full bg-sky-500 text-white flex items-center gap-1">
      <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
        <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" />
        <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
        <line x1="12" y1="19" x2="12" y2="23" />
      </svg>
      Transcript
    </span>
  );
}

type FilterMode = "all" | "transcript" | "visual";

/* ─── Component ──────────────────────────────────────────────────────── */

export default function VideoWorkspacePage() {
  /* refs */
  const inputRef = useRef<HTMLInputElement | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);

  /* file / local preview */
  const [file, setFile] = useState<File | null>(null);
  const [localUrl, setLocalUrl] = useState<string | null>(null);

  /* backend state */
  const [job, setJob] = useState<JobStatus | null>(null);
  const [polling, setPolling] = useState(false);

  /* search state */
  const [query, setQuery] = useState("");
  const [searching, setSearching] = useState(false);
  const [results, setResults] = useState<SearchResponse | null>(null);
  const [searchError, setSearchError] = useState<string | null>(null);

  /* filter mode */
  const [filterMode, setFilterMode] = useState<FilterMode>("all");

  /* active hit highlight */
  const [activeIdx, setActiveIdx] = useState<number | null>(null);

  /* clip end time - when set, video will pause at this time */
  const [clipEndTime, setClipEndTime] = useState<number | null>(null);

  /* ── Upload handler ──────────────────────────────────────────────── */
  const handleFileChange = useCallback(async (f: File | null) => {
    if (!f) return;
    setFile(f);
    setLocalUrl(URL.createObjectURL(f));
    setJob(null);
    setResults(null);
    setSearchError(null);
    setFilterMode("all");

    // Upload to backend
    const form = new FormData();
    form.append("file", f);
    try {
      const res = await fetch(`${API}/videos/upload`, { method: "POST", body: form });
      if (!res.ok) throw new Error(await res.text());
      const data: JobStatus = await res.json();
      setJob(data);
      setPolling(true);
    } catch (err: any) {
      setJob({ job_id: "", video_id: "", status: "failed", progress: 0, message: err.message });
    }
  }, []);

  /* ── Poll job status ─────────────────────────────────────────────── */
  useEffect(() => {
    if (!polling || !job?.job_id) return;

    const id = setInterval(async () => {
      try {
        const res = await fetch(`${API}/jobs/${job.job_id}`);
        if (!res.ok) return;
        const data: JobStatus = await res.json();
        setJob(data);
        if (data.status === "ready" || data.status === "failed") {
          setPolling(false);
        }
      } catch {
        /* network blip — keep polling */
      }
    }, 2000);

    return () => clearInterval(id);
  }, [polling, job?.job_id]);

  /* ── Stop video at clip end time ─────────────────────────────────── */
  useEffect(() => {
    const video = videoRef.current;
    if (!video || clipEndTime === null) return;

    const handleTimeUpdate = () => {
      if (video.currentTime >= clipEndTime) {
        video.pause();
        setClipEndTime(null);
      }
    };

    video.addEventListener("timeupdate", handleTimeUpdate);
    return () => video.removeEventListener("timeupdate", handleTimeUpdate);
  }, [clipEndTime]);

  /* ── Search handler ──────────────────────────────────────────────── */
  const handleSearch = useCallback(async () => {
    if (!job?.video_id || !query.trim()) return;
    setSearching(true);
    setSearchError(null);
    setResults(null);
    setActiveIdx(null);

    const trimmed = query.trim();

    try {
      // Check for clip: syntax (e.g., "clip: 5-10" or "clip:5-10")
      const clipMatch = trimmed.match(/^clip:\s*(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)$/i);
      if (clipMatch) {
        const t1 = parseFloat(clipMatch[1]);
        const t2 = parseFloat(clipMatch[2]);

        const res = await fetch(`${API}/videos/${job.video_id}/clip?t1=${t1}&t2=${t2}`);
        if (!res.ok) throw new Error(await res.text());
        const data = await res.json();

        setResults({
          query: trimmed,
          video_id: job.video_id,
          hits: [],
          best_window: {
            start_seconds: t1,
            end_seconds: t2,
            hit_count: 1,
          },
          clip_url: data.clip_url,
          visual_hits_count: 0,
          transcript_hits_count: 0,
        });

        // Seek video to the clip start and set end time
        if (videoRef.current) {
          videoRef.current.currentTime = t1;
          setClipEndTime(t2);
          videoRef.current.play().catch(() => {});
        }
      } else {
        // Normal semantic search
        const params = new URLSearchParams({
          q: trimmed,
          top_k: "10",
          include_visual: "true",
        });
        const res = await fetch(`${API}/videos/${job.video_id}/search?${params}`);
        if (!res.ok) throw new Error(await res.text());
        const data: SearchResponse = await res.json();
        setResults(data);
      }
    } catch (err: any) {
      setSearchError(err.message);
    } finally {
      setSearching(false);
    }
  }, [job?.video_id, query]);

  /* ── Seek video to a hit's timestamp ─────────────────────────────── */
  const seekTo = useCallback(
    (idx: number, startSec: number, endSec?: number) => {
      setActiveIdx(idx);
      if (videoRef.current) {
        videoRef.current.currentTime = startSec;
        if (endSec !== undefined) {
          setClipEndTime(endSec);
        } else {
          setClipEndTime(null);
        }
        videoRef.current.play().catch(() => {});
      }
    },
    [],
  );

  /* ── Derived state ──────────────────────────────────────────────── */
  const isReady = job?.status === "ready";
  const isProcessing = job?.status === "processing" || job?.status === "pending";
  const isFailed = job?.status === "failed";

  /* Filtered hits */
  const filteredHits = results?.hits.filter((h) => {
    if (filterMode === "all") return true;
    return h.source === filterMode;
  }) ?? [];

  const visualCount = results?.visual_hits_count ?? 0;
  const transcriptCount = results?.transcript_hits_count ?? 0;

  /* ─── Render ─────────────────────────────────────────────────────── */
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
      {/* ── Header ──────────────────────────────────────────────── */}
      <header className="bg-slate-900/80 backdrop-blur-sm border-b border-slate-700/50 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-violet-500 to-indigo-600 flex items-center justify-center">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2">
                <polygon points="5 3 19 12 5 21 5 3" />
              </svg>
            </div>
            <div>
              <h1 className="text-white font-semibold text-lg">CortexON</h1>
              <p className="text-slate-400 text-xs">Video Understanding + CLIP Visual Search</p>
            </div>
          </div>
          
          <label className="cursor-pointer">
            <input
              type="file"
              accept="video/*"
              className="hidden"
              onChange={(e) => handleFileChange(e.target.files?.[0] ?? null)}
            />
            <span className="bg-gradient-to-r from-violet-600 to-indigo-600 hover:from-violet-500 hover:to-indigo-500 
                           text-white text-sm font-medium px-4 py-2 rounded-lg transition-all shadow-lg shadow-violet-500/25">
              Upload Video
            </span>
          </label>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
          
          {/* ── Left: Video Panel ──────────────────────────────────── */}
          <div className="lg:col-span-3 space-y-4">
            {/* Video Player */}
            <div className="bg-slate-800/50 rounded-2xl overflow-hidden border border-slate-700/50 shadow-xl">
              {localUrl ? (
                <video
                  ref={videoRef}
                  src={localUrl}
                  controls
                  className="w-full aspect-video bg-black"
                />
              ) : (
                <div className="aspect-video bg-slate-900/50 flex flex-col items-center justify-center text-slate-500">
                  <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1">
                    <rect x="2" y="2" width="20" height="20" rx="2.18" ry="2.18" />
                    <line x1="7" y1="2" x2="7" y2="22" />
                    <line x1="17" y1="2" x2="17" y2="22" />
                    <line x1="2" y1="12" x2="22" y2="12" />
                    <line x1="2" y1="7" x2="7" y2="7" />
                    <line x1="2" y1="17" x2="7" y2="17" />
                    <line x1="17" y1="17" x2="22" y2="17" />
                    <line x1="17" y1="7" x2="22" y2="7" />
                  </svg>
                  <p className="mt-4 text-sm">Upload a video to get started</p>
                </div>
              )}
            </div>

            {/* Status Bar */}
            {job && (
              <div className={`rounded-xl p-4 border ${
                isFailed ? "bg-red-900/20 border-red-500/30" :
                isProcessing ? "bg-amber-900/20 border-amber-500/30" :
                "bg-emerald-900/20 border-emerald-500/30"
              }`}>
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    {isProcessing && (
                      <div className="w-5 h-5 border-2 border-amber-500 border-t-transparent rounded-full animate-spin" />
                    )}
                    {isReady && (
                      <div className="w-5 h-5 rounded-full bg-emerald-500 flex items-center justify-center">
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="3">
                          <polyline points="20 6 9 17 4 12" />
                        </svg>
                      </div>
                    )}
                    {isFailed && (
                      <div className="w-5 h-5 rounded-full bg-red-500 flex items-center justify-center">
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="3">
                          <line x1="18" y1="6" x2="6" y2="18" />
                          <line x1="6" y1="6" x2="18" y2="18" />
                        </svg>
                      </div>
                    )}
                    <span className={`text-sm font-medium ${
                      isFailed ? "text-red-300" : isProcessing ? "text-amber-300" : "text-emerald-300"
                    }`}>
                      {job.message}
                    </span>
                  </div>
                  {isProcessing && (
                    <span className="text-amber-400 text-sm font-mono">
                      {Math.round(job.progress * 100)}%
                    </span>
                  )}
                </div>
                {isProcessing && (
                  <div className="mt-3 h-1.5 bg-slate-700 rounded-full overflow-hidden">
                    <div 
                      className="h-full bg-gradient-to-r from-amber-500 to-amber-400 transition-all duration-500"
                      style={{ width: `${job.progress * 100}%` }}
                    />
                  </div>
                )}
              </div>
            )}

            {/* Search Box */}
            {isReady && (
              <div className="bg-slate-800/50 rounded-xl p-4 border border-slate-700/50">
                <div className="flex gap-3">
                  <input
                    ref={inputRef}
                    type="text"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    onKeyDown={(e) => e.key === "Enter" && handleSearch()}
                    placeholder="Search video content... (or use clip: 5-10 for direct timestamps)"
                    className="flex-1 bg-slate-900/50 border border-slate-600/50 rounded-lg px-4 py-3 text-white 
                             placeholder-slate-500 focus:outline-none focus:border-violet-500/50 focus:ring-2 
                             focus:ring-violet-500/20 transition-all"
                  />
                  <button
                    onClick={handleSearch}
                    disabled={searching || !query.trim()}
                    className="bg-gradient-to-r from-violet-600 to-indigo-600 hover:from-violet-500 hover:to-indigo-500
                             disabled:from-slate-600 disabled:to-slate-600 disabled:cursor-not-allowed
                             text-white font-medium px-6 py-3 rounded-lg transition-all shadow-lg shadow-violet-500/25
                             disabled:shadow-none flex items-center gap-2"
                  >
                    {searching ? (
                      <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                    ) : (
                      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <circle cx="11" cy="11" r="8" />
                        <path d="m21 21-4.35-4.35" />
                      </svg>
                    )}
                    Search
                  </button>
                </div>
                <p className="text-slate-500 text-xs mt-2">
                  💡 Try visual queries like "person walking", "red car", or "outdoor scene"
                </p>
              </div>
            )}
          </div>

          {/* ── Right: Results Panel ───────────────────────────────── */}
          <div className="lg:col-span-2">
            <div className="bg-slate-800/50 rounded-2xl border border-slate-700/50 overflow-hidden sticky top-20">
              {/* Results Header */}
              <div className="bg-slate-900/50 px-4 py-3 border-b border-slate-700/50">
                <h2 className="text-white font-semibold flex items-center gap-2">
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <circle cx="11" cy="11" r="8" />
                    <path d="m21 21-4.35-4.35" />
                  </svg>
                  Search Results
                </h2>
                {results && (
                  <p className="text-slate-400 text-xs mt-1">
                    {transcriptCount} transcript + {visualCount} visual matches
                  </p>
                )}
              </div>

              {/* Filter Tabs */}
              {results && results.hits.length > 0 && (
                <div className="flex gap-1 p-2 bg-slate-900/30 border-b border-slate-700/50">
                  {[
                    { mode: "all" as FilterMode, label: `All (${results.hits.length})` },
                    { mode: "transcript" as FilterMode, label: `🎙️ Transcript (${transcriptCount})` },
                    { mode: "visual" as FilterMode, label: `👁️ Visual (${visualCount})` },
                  ].map(({ mode, label }) => (
                    <button
                      key={mode}
                      onClick={() => setFilterMode(mode)}
                      className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                        filterMode === mode
                          ? "bg-violet-600 text-white shadow-lg"
                          : "text-slate-400 hover:text-white hover:bg-slate-700/50"
                      }`}
                    >
                      {label}
                    </button>
                  ))}
                </div>
              )}

              {/* Results Content */}
              <div className="max-h-[calc(100vh-280px)] overflow-y-auto p-4 space-y-3">
                {searchError && (
                  <div className="bg-red-900/20 border border-red-500/30 rounded-lg p-3 text-red-300 text-sm">
                    {searchError}
                  </div>
                )}

                {/* Best Window */}
                {results?.best_window && (
                  <div
                    className="bg-gradient-to-r from-violet-600/20 to-indigo-600/20 rounded-xl p-4 
                             border border-violet-500/30 cursor-pointer hover:border-violet-500/50 
                             transition-all group"
                    onClick={() => seekTo(-1, results.best_window!.start_seconds, results.best_window!.end_seconds)}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-violet-300 text-xs font-semibold uppercase tracking-wider">
                        Best Match Window
                      </span>
                      <span className="text-violet-400 text-xs opacity-0 group-hover:opacity-100 transition-opacity">
                        Click to play →
                      </span>
                    </div>
                    <div className="flex items-center gap-3">
                      <span className="text-white font-mono text-lg">
                        {fmtTime(results.best_window.start_seconds)} – {fmtTime(results.best_window.end_seconds)}
                      </span>
                      <span className="text-slate-400 text-sm">
                        ({results.best_window.hit_count} chunk{results.best_window.hit_count !== 1 ? "s" : ""})
                      </span>
                    </div>
                    {results.clip_url && (
                      <a
                        href={results.clip_url}
                        download
                        onClick={(e) => e.stopPropagation()}
                        className="inline-flex items-center gap-1.5 mt-3 text-violet-400 hover:text-violet-300 
                                 text-sm transition-colors"
                      >
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                          <polyline points="7 10 12 15 17 10" />
                          <line x1="12" y1="15" x2="12" y2="3" />
                        </svg>
                        Download clip
                      </a>
                    )}
                  </div>
                )}

                {/* Individual Hits */}
                {filteredHits.map((hit, i) => (
                  <div
                    key={i}
                    className={`rounded-xl p-4 cursor-pointer transition-all border ${
                      activeIdx === i
                        ? "bg-violet-600/20 border-violet-500/50 shadow-lg shadow-violet-500/10"
                        : hit.source === "visual"
                        ? "bg-violet-900/10 border-violet-500/20 hover:bg-violet-900/20 hover:border-violet-500/30"
                        : "bg-slate-700/20 border-slate-600/30 hover:bg-slate-700/30 hover:border-slate-500/30"
                    }`}
                    onClick={() => seekTo(i, hit.start_seconds, hit.end_seconds)}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <span className="font-mono text-white text-sm">
                          {fmtTime(hit.start_seconds)} – {fmtTime(hit.end_seconds)}
                        </span>
                        <SourceBadge source={hit.source} />
                      </div>
                      <MatchBadge distance={hit.distance} />
                    </div>
                    <p className={`text-sm leading-relaxed line-clamp-3 ${
                      hit.source === "visual" 
                        ? "text-violet-200/70 italic" 
                        : "text-slate-300"
                    }`}>
                      {hit.text || (hit.source === "visual" ? "[Visual match - no description]" : "[No text]")}
                    </p>
                  </div>
                ))}

                {/* Empty State */}
                {!results && !searchError && isReady && (
                  <div className="text-center py-12">
                    <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-slate-700/50 flex items-center justify-center">
                      <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="text-slate-500">
                        <circle cx="11" cy="11" r="8" />
                        <path d="m21 21-4.35-4.35" />
                      </svg>
                    </div>
                    <p className="text-slate-400 text-sm">Enter a query to search the video</p>
                    <p className="text-slate-500 text-xs mt-1">
                      Visual search powered by CLIP
                    </p>
                  </div>
                )}

                {results && filteredHits.length === 0 && (
                  <div className="text-center py-8">
                    <p className="text-slate-400 text-sm">No matches found for this filter</p>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

