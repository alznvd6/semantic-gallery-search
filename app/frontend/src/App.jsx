import { useCallback, useEffect, useMemo, useRef, useState } from "react";

const CATEGORIES = ["آدم‌ها", "مکان‌ها", "سفر", "طبیعت", "خوراک و نوشیدنی"];
const PAGE_SIZE = 24;
const SEARCH_STAGES = [
  "توضیح شما به یک بردار معنایی تبدیل می‌شود…",
  "مدل در حال بررسی گالری شماست…",
  "در حال مقایسه‌ی توضیح شما با عکس‌های آرشیو…",
];

async function readResponse(response) {
  const data = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = data && data.detail;
    throw new Error(
      typeof detail === "string" ? detail : `درخواست ناموفق بود (${response.status}).`,
    );
  }
  return data;
}

async function fetchJson(url, options) {
  return readResponse(await fetch(url, options));
}

const fa = (value) => Number(value).toLocaleString("fa-IR");

function titleFromFilename(filename) {
  const base = String(filename || "").replace(/\.[^.]+$/, "");
  const numbered = base.match(/^img[_-]?(\d+)$/i);
  if (numbered) return `عکس ${Number(numbered[1]).toLocaleString("fa-IR")}`;
  return base.replace(/[-_]+/g, " ").trim() || filename;
}

function formatDate(value) {
  if (!value) return "";
  const date = new Date(`${value}T12:00:00`);
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleDateString("fa-IR", {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
}

function normalizePhoto(photo) {
  return {
    id: photo.filename,
    filename: photo.filename,
    image_url: photo.image_url,
    date: photo.date || null,
    title: titleFromFilename(photo.filename),
    tag: "آرشیو شما",
    place: "آرشیو شخصی",
    bio: "از آرشیو شخصی شما",
    preview: false,
  };
}

function CameraIcon() {
  return (
    <svg viewBox="0 0 48 48" fill="none" aria-hidden="true">
      <path
        d="M8 16.5h7l3-4h12l3 4h7a3 3 0 0 1 3 3v16a3 3 0 0 1-3 3H8a3 3 0 0 1-3-3v-16a3 3 0 0 1 3-3Z"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinejoin="round"
      />
      <path
        d="M18 12.5h12M10 21h4"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
        opacity=".8"
      />
      <circle cx="24" cy="27.5" r="8" stroke="currentColor" strokeWidth="1.8" />
      <circle cx="24" cy="27.5" r="4.3" stroke="currentColor" strokeWidth="1.5" />
      <circle cx="36" cy="21.5" r="1.5" fill="currentColor" />
    </svg>
  );
}

function SearchIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <circle cx="11" cy="11" r="7" />
      <path d="m20 20-4-4" />
    </svg>
  );
}

function LibraryIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <rect x="3" y="3" width="8" height="8" rx="1.5" />
      <rect x="13" y="3" width="8" height="8" rx="1.5" />
      <rect x="3" y="13" width="8" height="8" rx="1.5" />
      <rect x="13" y="13" width="8" height="8" rx="1.5" />
    </svg>
  );
}

function AlbumIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <rect x="3" y="5" width="18" height="15" rx="2" />
      <path d="M3 9h18M8 5V3h8v2" />
    </svg>
  );
}

function CategoryIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M4 5h7v7H4zM14 5h6v4h-6zM14 12h6v7h-6zM4 15h7v4H4z" />
    </svg>
  );
}

function MicrophoneIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <rect x="9" y="3" width="6" height="12" rx="3" />
      <path d="M5 11a7 7 0 0 0 14 0M12 18v3" />
    </svg>
  );
}

function UploadIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <rect x="3" y="5" width="14" height="14" rx="2" />
      <circle cx="8" cy="10" r="1.5" />
      <path d="m3 16 4-4 4 4 2-2 4 4" />
      <path d="M19 3v6M16 6h6" />
    </svg>
  );
}

function FolderIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" />
      <path d="M12 11v5M9.5 13.5h5" />
    </svg>
  );
}

function IndexIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="m12 3 8 4.5-8 4.5-8-4.5L12 3Z" />
      <path d="m4 12 8 4.5 8-4.5" />
      <path d="m4 16.5 8 4.5 8-4.5" />
    </svg>
  );
}

function PhotoCard({ photo, onOpen }) {
  return (
    <article
      className="photo-card"
      tabIndex={0}
      role="button"
      aria-label={`نمایش ${photo.title}`}
      onClick={onOpen}
      onKeyDown={(event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          onOpen();
        }
      }}
    >
      <div className="photo-wrap">
        <img src={photo.image_url} alt={photo.title} loading="lazy" />
        <span className="photo-tag">{photo.tag}</span>
      </div>
      {photo.score !== undefined && (
        <div className="match">
          <span>شباهت به جستجو</span>
          <b>{fa(Math.max(0, Math.round(photo.score * 100)))}٪</b>
        </div>
      )}
      <div className="photo-info">
        <h3 className="photo-title">{photo.title}</h3>
        <p className="photo-bio">{photo.bio}</p>
        <div className="photo-meta">
          <span>⌖ {photo.place}</span>
          <span>{formatDate(photo.date) || "بدون تاریخ"}</span>
        </div>
      </div>
    </article>
  );
}

function Gallery({ items, feed, onOpen }) {
  return (
    <div className={`gallery${feed ? " feed" : ""}`}>
      {items.map((photo, index) => (
        <PhotoCard key={`${photo.id}-${index}`} photo={photo} onOpen={() => onOpen(index)} />
      ))}
    </div>
  );
}

function LoadingPanel({ title, message, note }) {
  return (
    <div className="empty search-status" role="status" aria-live="polite">
      <span className="spinner" aria-hidden="true" />
      <strong>{title}</strong>
      {message && <p>{message}</p>}
      {note && <small>{note}</small>}
    </div>
  );
}

function EmptyState({ title, children, note, action }) {
  return (
    <div className="empty" role="status">
      <strong>{title}</strong>
      {children && <p>{children}</p>}
      {note && <small>{note}</small>}
      {action && <div className="empty-actions">{action}</div>}
    </div>
  );
}

function App() {
  const [photos, setPhotos] = useState([]);
  const [previews, setPreviews] = useState([]);
  const [galleryLoading, setGalleryLoading] = useState(true);
  const [galleryError, setGalleryError] = useState("");
  const [health, setHealth] = useState(null);

  const [page, setPage] = useState("search");
  const [query, setQuery] = useState("");
  const [search, setSearch] = useState({ status: "idle", query: "", results: [], error: "" });
  const [semantic, setSemantic] = useState({ term: "", status: "idle", results: [], error: "" });
  const [stage, setStage] = useState(0);

  const [activeFilter, setActiveFilter] = useState("All");
  const [selectedCategory, setSelectedCategory] = useState("All");
  const [visibleCount, setVisibleCount] = useState(PAGE_SIZE);
  const [lightboxIndex, setLightboxIndex] = useState(null);

  const [expandedNav, setExpandedNav] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [collapsed, setCollapsed] = useState(false);
  const [listening, setListening] = useState(false);
  const [toastMessage, setToastMessage] = useState("");
  const [indexStarting, setIndexStarting] = useState(false);

  const fileInput = useRef(null);
  const folderInput = useRef(null);
  const lightbox = useRef(null);
  const toastTimer = useRef(null);
  const previewUrls = useRef(new Set());
  const searchRequest = useRef(0);
  const semanticRequest = useRef(0);
  const wasIndexing = useRef(false);

  const toast = useCallback((message) => {
    setToastMessage(message);
    clearTimeout(toastTimer.current);
    toastTimer.current = setTimeout(() => setToastMessage(""), 3200);
  }, []);

  const refreshHealth = useCallback(async () => {
    try {
      const data = await fetchJson("/api/health");
      setHealth(data);
    } catch {
      /* keep the last known status when the backend is unreachable */
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();

    (async () => {
      try {
        const data = await fetchJson("/api/gallery", { signal: controller.signal });
        setPhotos((data.photos || []).map(normalizePhoto));
      } catch (error) {
        if (error.name !== "AbortError") {
          setGalleryError(
            error instanceof Error ? error.message : "ارتباط با سرویس گالری برقرار نشد.",
          );
        }
      } finally {
        if (!controller.signal.aborted) setGalleryLoading(false);
      }
    })();

    refreshHealth();
    return () => controller.abort();
  }, [refreshHealth]);

  useEffect(
    () => () => {
      clearTimeout(toastTimer.current);
      previewUrls.current.forEach((url) => URL.revokeObjectURL(url));
      previewUrls.current.clear();
    },
    [],
  );

  const allPhotos = useMemo(() => [...previews, ...photos], [photos, previews]);
  const photosByName = useMemo(
    () => new Map(allPhotos.map((photo) => [photo.filename, photo])),
    [allPhotos],
  );

  const searchBusy = search.status === "loading";
  const semanticBusy = semantic.status === "loading";
  const busy = searchBusy || semanticBusy;

  useEffect(() => {
    if (!busy) {
      setStage(0);
      return undefined;
    }
    const timer = setInterval(
      () => setStage((current) => (current + 1) % SEARCH_STAGES.length),
      6000,
    );
    return () => clearInterval(timer);
  }, [busy]);

  useEffect(() => {
    if (!busy) return undefined;
    const timer = setInterval(refreshHealth, 4000);
    return () => clearInterval(timer);
  }, [busy, refreshHealth]);

  useEffect(() => {
    if (!health?.indexing) {
      if (!wasIndexing.current) return undefined;
      wasIndexing.current = false;
      const ready = Boolean(health?.index_ready) && !health?.index_stale;
      toast(
        ready
          ? "نمایه‌ی گالری ساخته و ذخیره شد."
          : "ساخت نمایه کامل نشد؛ گزارش سرور را بررسی کنید.",
      );
      return undefined;
    }

    wasIndexing.current = true;
    const timer = setInterval(refreshHealth, 2500);
    return () => clearInterval(timer);
  }, [health, refreshHealth, toast]);

  function toResult(result) {
    const base = photosByName.get(result.filename);
    return {
      ...(base || normalizePhoto(result)),
      id: result.filename,
      score: result.score,
      tag: "نتیجه‌ی جستجو",
    };
  }

  function toResults(data) {
    return (data.results || [])
      .map(toResult)
      .sort((a, b) => (b.score ?? 0) - (a.score ?? 0));
  }

  async function runSearch(text) {
    const normalized = text.trim();
    if (!normalized) {
      toast("برای جستجو، چیزی درباره‌ی عکس بنویسید.");
      return;
    }

    const token = ++searchRequest.current;
    setPage("search");
    setQuery(normalized);
    setLightboxIndex(null);
    setSearch({ status: "loading", query: normalized, results: [], error: "" });

    try {
      const data = await fetchJson("/api/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: normalized, limit: 30 }),
      });
      if (searchRequest.current !== token) return;
      const results = toResults(data);
      setSearch({ status: "done", query: normalized, results, error: "" });
      setHealth((current) => (current ? { ...current, index_ready: true } : current));
      refreshHealth();
      if (!results.length) toast("عکسی با این توضیح پیدا نشد.");
    } catch (error) {
      if (searchRequest.current !== token) return;
      const message =
        error instanceof Error ? error.message : "جستجو انجام نشد. اتصال سرویس را بررسی کنید.";
      setSearch({ status: "error", query: normalized, results: [], error: message });
      toast("جستجو انجام نشد.");
      refreshHealth();
    }
  }

  async function loadSemantic(term) {
    const token = ++semanticRequest.current;
    setSemantic({ term, status: "loading", results: [], error: "" });
    setLightboxIndex(null);

    try {
      const data = await fetchJson("/api/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: term, limit: 30 }),
      });
      if (semanticRequest.current !== token) return;
      setSemantic({ term, status: "done", results: toResults(data), error: "" });
      setHealth((current) => (current ? { ...current, index_ready: true } : current));
      refreshHealth();
    } catch (error) {
      if (semanticRequest.current !== token) return;
      const message =
        error instanceof Error ? error.message : "بررسی این مجموعه ممکن نشد.";
      setSemantic({ term, status: "error", results: [], error: message });
      toast("بررسی گالری ممکن نشد.");
      refreshHealth();
    }
  }

  async function startIndexBuild() {
    if (indexStarting || health?.indexing) return;
    setIndexStarting(true);

    try {
      const data = await fetchJson("/api/index", { method: "POST" });
      setHealth((current) => (current ? { ...current, indexing: true } : current));
      toast(
        `ساخت نمایه برای ${fa(data.photo_count)} عکس شروع شد؛ تا پایان صبر کنید.`,
      );
    } catch (error) {
      toast(
        error instanceof Error
          ? error.message
          : "شروع ساخت نمایه ممکن نشد. اتصال سرویس را بررسی کنید.",
      );
    } finally {
      setIndexStarting(false);
      refreshHealth();
    }
  }

  function goTo(nextPage) {
    searchRequest.current += 1;
    semanticRequest.current += 1;
    setPage(nextPage);
    setQuery("");
    setSearch({ status: "idle", query: "", results: [], error: "" });
    setSemantic({ term: "", status: "idle", results: [], error: "" });
    setActiveFilter("All");
    setSelectedCategory("All");
    setVisibleCount(PAGE_SIZE);
    setLightboxIndex(null);
    setSidebarOpen(false);
  }

  function openCategory(name) {
    setPage("categories");
    setSearch({ status: "idle", query: "", results: [], error: "" });
    setSelectedCategory(name);
    setLightboxIndex(null);
    setSidebarOpen(false);
    loadSemantic(name);
  }

  function chooseFilter(category) {
    setActiveFilter(category);
    setVisibleCount(PAGE_SIZE);
    setLightboxIndex(null);
    if (category !== "All") {
      loadSemantic(category);
    } else {
      semanticRequest.current += 1;
      setSemantic({ term: "", status: "idle", results: [], error: "" });
    }
  }

  function addLocalFiles(fileList) {
    const files = Array.from(fileList || []).filter((file) =>
      file.type.startsWith("image/"),
    );
    if (!files.length) {
      toast("در فایل‌های انتخاب‌شده تصویری پیدا نشد.");
      return;
    }

    const added = files.map((file) => {
      const url = URL.createObjectURL(file);
      previewUrls.current.add(url);
      return {
        id: url,
        filename: file.name,
        image_url: url,
        title: titleFromFilename(file.name),
        bio: "فقط در این مرورگر؛ در آرشیو ذخیره نشده است.",
        place: "این دستگاه",
        tag: "پیش‌نمایش",
        date: new Date().toISOString().slice(0, 10),
        preview: true,
      };
    });

    setPreviews((current) => [...added, ...current]);
    goTo("library");
    toast(
      `${fa(files.length)} عکس برای پیش‌نمایش اضافه شد؛ برای ذخیره و جستجو، فایل‌ها را در پوشه‌ی data/images بگذارید و دکمه‌ی «ساخت نمایه» را بزنید.`,
    );
  }

  function startVoiceSearch() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      toast("مرورگر شما از جستجوی صوتی پشتیبانی نمی‌کند.");
      return;
    }

    const recognition = new SpeechRecognition();
    recognition.lang = "fa-IR";
    recognition.interimResults = false;
    recognition.onstart = () => setListening(true);
    recognition.onresult = (event) => {
      const transcript = event.results[0][0].transcript;
      setQuery(transcript);
      runSearch(transcript);
    };
    recognition.onerror = () => {
      setListening(false);
      toast("دریافت صدا ممکن نشد. دوباره تلاش کنید.");
    };
    recognition.onend = () => setListening(false);

    try {
      recognition.start();
    } catch {
      setListening(false);
      toast("شروع جستجوی صوتی ممکن نشد.");
    }
  }

  function handleSubmit(event) {
    event.preventDefault();
    runSearch(query);
  }

  useEffect(() => {
    function handleGlobalKeys(event) {
      if (event.key === "Escape") setSidebarOpen(false);
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        goTo("search");
        requestAnimationFrame(() => document.getElementById("photo-search")?.focus());
      }
    }
    window.addEventListener("keydown", handleGlobalKeys);
    return () => window.removeEventListener("keydown", handleGlobalKeys);
  }, []);

  const displayed = useMemo(() => {
    if (page === "search") {
      if (search.status === "done") return search.results;
      return allPhotos.slice(0, 6);
    }
    if (page === "library") {
      if (activeFilter === "All") return allPhotos.slice(0, visibleCount);
      if (semantic.term === activeFilter && semantic.status === "done") return semantic.results;
      return [];
    }
    if (page === "categories" && selectedCategory !== "All") {
      if (semantic.term === selectedCategory && semantic.status === "done") {
        return semantic.results;
      }
    }
    return [];
  }, [activeFilter, allPhotos, page, search, selectedCategory, semantic, visibleCount]);

  const currentPhoto = lightboxIndex === null ? null : displayed[lightboxIndex];

  useEffect(() => {
    if (!currentPhoto) return undefined;
    document.body.style.overflow = "hidden";

    function handleKeys(event) {
      if (event.key === "Escape") {
        if (!document.fullscreenElement) setLightboxIndex(null);
        else document.exitFullscreen().catch(() => {});
      } else if (event.key === "ArrowLeft") {
        setLightboxIndex((index) => (index + 1) % displayed.length);
      } else if (event.key === "ArrowRight") {
        setLightboxIndex((index) => (index - 1 + displayed.length) % displayed.length);
      } else if (event.key.toLowerCase() === "f") {
        toggleFullscreen();
      }
    }

    function toggleFullscreen() {
      if (document.fullscreenElement) {
        document.exitFullscreen().catch(() => {});
        return;
      }
      const request = lightbox.current?.requestFullscreen;
      if (request) request.call(lightbox.current).catch(() => {});
    }

    window.addEventListener("keydown", handleKeys);
    return () => {
      document.body.style.overflow = "";
      window.removeEventListener("keydown", handleKeys);
    };
  }, [currentPhoto, displayed.length]);

  const breadcrumb = {
    search: "جستجوی متنی",
    library: "همه‌ی تصاویر",
    albums: "آلبوم‌ها",
    categories: selectedCategory === "All" ? "دسته‌بندی‌ها" : selectedCategory,
  }[page];

  const isFeed = page === "search" && search.status === "idle";
  const indexReady = Boolean(health?.index_ready);
  const indexRunning = Boolean(health?.indexing);
  const indexOutdated =
    indexReady &&
    (Boolean(health?.index_stale) ||
      (health?.indexed_count != null &&
        health?.photo_count != null &&
        health.indexed_count !== health.photo_count));
  const indexLabel = indexRunning
    ? "در حال ساخت نمایه…"
    : indexStarting
      ? "در حال شروع…"
      : !indexReady
        ? "ساخت نمایه"
        : indexOutdated
          ? "به‌روزرسانی نمایه"
          : "بازسازی نمایه";
  const modelNote = indexReady
    ? ""
    : "اولین جستجو مدل Jina CLIP v2 را بارگذاری و نمایه‌ی گالری را می‌سازد؛ ممکن است چند دقیقه طول بکشد.";

  const modelStatus = (() => {
    if (galleryLoading) return { text: "در حال خواندن گالری…", tone: "busy" };
    if (galleryError && !allPhotos.length) return { text: "سرویس در دسترس نیست", tone: "warn" };
    if (!allPhotos.length) return { text: "گالری خالی است", tone: "warn" };
    if (busy || health?.indexing) return { text: "مدل در حال بررسی گالری…", tone: "busy" };
    if (!health) return { text: "وضعیت مدل نامعلوم", tone: "warn" };
    if (!health.index_ready) return { text: "نمایه ساخته نشده", tone: "warn" };
    if (indexOutdated) return { text: "نمایه قدیمی است", tone: "warn" };
    return { text: "مدل آماده است", tone: "ok" };
  })();

  const galleryNotice = galleryLoading ? (
    <LoadingPanel title="در حال خواندن عکس‌های گالری…" message="کمی صبر کنید…" />
  ) : galleryError && !allPhotos.length ? (
    <EmptyState
      title="ارتباط با سرویس برقرار نشد."
      note="بک‌اند را با run_app_backend.ps1 اجرا کنید و صفحه را دوباره باز کنید."
    >
      {galleryError}
    </EmptyState>
  ) : !allPhotos.length ? (
    <EmptyState
      title="هنوز عکسی در گالری نیست."
      note="عکس‌ها را در پوشه‌ی data/images قرار دهید و صفحه را دوباره باز کنید."
    />
  ) : null;

  return (
    <div className={`app${collapsed ? " collapsed" : ""}`}>
      {sidebarOpen && (
        <button
          className="sidebar-backdrop"
          type="button"
          aria-label="بستن منو"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      <aside className={`sidebar${sidebarOpen ? " open" : ""}`} id="sidebar">
        <a
          className="brand"
          href="#"
          aria-label="صفحه‌ی اصلی گالری معنایی"
          onClick={(event) => {
            event.preventDefault();
            goTo("search");
          }}
        >
          <span className="brand-icon" aria-hidden="true">
            <CameraIcon />
          </span>
          <span className="brand-name">
            Semantic
            <br />
            Gallery
          </span>
        </a>

        <nav className="nav" aria-label="ناوبری اصلی">
          <button
            className={page === "search" ? "active" : ""}
            type="button"
            title="جستجوی متنی"
            onClick={() => goTo("search")}
          >
            <SearchIcon />
            <span className="nav-text">جستجوی متنی</span>
          </button>
          <button
            className={page === "library" ? "active" : ""}
            type="button"
            title="همه‌ی تصاویر"
            onClick={() => goTo("library")}
          >
            <LibraryIcon />
            <span className="nav-text">همه‌ی تصاویر</span>
          </button>
          <button
            className={page === "albums" ? "active" : ""}
            type="button"
            title="آلبوم‌ها"
            onClick={() => goTo("albums")}
          >
            <AlbumIcon />
            <span className="nav-text">آلبوم‌ها</span>
          </button>
          <button
            className={page === "categories" ? "active" : ""}
            type="button"
            title="دسته‌بندی‌ها"
            aria-expanded={expandedNav === "categories"}
            onClick={() => goTo("categories")}
          >
            <CategoryIcon />
            <span className="nav-text">دسته‌بندی‌ها</span>
            <span
              className="nav-expand"
              role="button"
              tabIndex={0}
              aria-label="باز کردن دسته‌بندی‌ها"
              onClick={(event) => {
                event.stopPropagation();
                setExpandedNav((current) => (current === "categories" ? "" : "categories"));
              }}
              onKeyDown={(event) => {
                if (event.key === "Enter" || event.key === " ") {
                  event.preventDefault();
                  event.stopPropagation();
                  setExpandedNav((current) => (current === "categories" ? "" : "categories"));
                }
              }}
            >
              ⌄
            </span>
          </button>
          <div className={`subnav${expandedNav === "categories" ? " open" : ""}`}>
            {CATEGORIES.map((category) => (
              <button
                key={category}
                type="button"
                className={selectedCategory === category ? "active" : ""}
                onClick={() => openCategory(category)}
              >
                {category}
              </button>
            ))}
          </div>
        </nav>

        <button
          className="icon-btn side-toggle"
          type="button"
          aria-label={collapsed ? "باز کردن نوار کناری" : "جمع‌کردن نوار کناری"}
          aria-expanded={!collapsed}
          title="جمع‌کردن / باز کردن نوار کناری"
          onClick={() => setCollapsed((current) => !current)}
        >
          <svg viewBox="0 0 24 24" aria-hidden="true">
            <rect x="3" y="4" width="18" height="16" rx="3" />
            <path d="M15 4v16" />
            <path className="chev" d="m9 9 3 3-3 3" />
          </svg>
        </button>
      </aside>

      <main className="main">
        <header className="topbar">
          <div className="topbar-start">
            <div className="crumb">
              <span className="crumb-camera" aria-hidden="true" />
              آرشیو شما
              <span style={{ padding: "0 7px", color: "#c2c7ca" }}>/</span>
              <b id="breadcrumb">{breadcrumb}</b>
            </div>
          </div>
          <div className="top-actions">
            <button
              className="index-btn"
              type="button"
              aria-label={indexLabel}
              title="همه‌ی عکس‌های پوشه‌ی گالری را تحلیل و بردارهایشان را ذخیره می‌کند. اگر نمایه ساخته نشده باشد، اولین جستجو هم این کار را خودکار انجام می‌دهد."
              disabled={indexRunning || indexStarting}
              onClick={startIndexBuild}
            >
              {indexRunning || indexStarting ? (
                <span className="mini-spinner" aria-hidden="true" />
              ) : (
                <IndexIcon />
              )}
              <span className="index-btn-text">{indexLabel}</span>
            </button>
            <span className={`model-status ${modelStatus.tone}`} title={modelStatus.text}>
              <span className="dot" aria-hidden="true" />
              {modelStatus.text}
            </span>
            <button
              className="icon-btn mobile-menu"
              type="button"
              aria-label="باز کردن منو"
              onClick={() => setSidebarOpen(true)}
            >
              <svg viewBox="0 0 24 24" aria-hidden="true">
                <path d="M4 7h16M4 12h16M4 17h16" />
              </svg>
            </button>
            <button
              className="upload sq"
              type="button"
              aria-label="افزودن عکس"
              title="افزودن عکس"
              onClick={() => fileInput.current?.click()}
            >
              <UploadIcon />
            </button>
            <button
              className="icon-btn folder-btn"
              type="button"
              aria-label="افزودن پوشه‌ای از عکس‌ها"
              title="افزودن پوشه‌ای از عکس‌ها"
              onClick={() => folderInput.current?.click()}
            >
              <FolderIcon />
            </button>
            <input
              ref={fileInput}
              type="file"
              accept="image/*"
              multiple
              hidden
              onChange={(event) => {
                addLocalFiles(event.target.files);
                event.target.value = "";
              }}
            />
            <input
              ref={folderInput}
              type="file"
              accept="image/*"
              multiple
              hidden
              webkitdirectory=""
              directory=""
              onChange={(event) => {
                addLocalFiles(event.target.files);
                event.target.value = "";
              }}
            />
          </div>
        </header>

        {page === "search" && (
          <>
            <section className="page-intro">
              <h1>
                خاطراتتو <em>گم نکن!</em>
              </h1>
            </section>
            <form className="search-row" onSubmit={handleSubmit}>
              <label className="searchbox" htmlFor="photo-search">
                <SearchIcon />
                <input
                  id="photo-search"
                  dir="rtl"
                  value={query}
                  maxLength={1000}
                  onChange={(event) => setQuery(event.target.value)}
                  placeholder="جستجوی سوژه، مکان، حس‌وحال یا لحظه…"
                />
                <span style={{ fontSize: 10, color: "#a1aab0" }} aria-hidden="true">
                  ↵
                </span>
              </label>
              <button
                className={`icon-btn mic-btn${listening ? " listening" : ""}`}
                type="button"
                aria-label="جستجوی صوتی"
                title="جستجوی صوتی"
                onClick={startVoiceSearch}
              >
                <MicrophoneIcon />
              </button>
              <button className="search-submit" type="submit" disabled={searchBusy}>
                {searchBusy ? "در حال جستجو…" : "جستجوی تصاویر"}
              </button>
            </form>

            <div className={`section-head${isFeed ? " slim" : ""}`}>
              <div>
                <h2>{isFeed ? "تازه‌ها در آرشیو شما" : "قاب‌های مرتبط"}</h2>
                <p>
                  {isFeed
                    ? "نگاهی به مجموعه‌ی عکس‌های شما"
                    : `عکس‌های شما که با «${search.query}» می‌خوانند`}
                </p>
              </div>
              <span style={{ fontSize: 12, color: "var(--muted)" }}>
                {fa(isFeed ? allPhotos.length : search.results.length)} عکس
              </span>
            </div>

            {searchBusy ? (
              <LoadingPanel
                title={indexReady ? "در حال جستجو در گالری…" : "مدل در حال بررسی گالری…"}
                message={SEARCH_STAGES[stage]}
                note={modelNote}
              />
            ) : search.status === "error" ? (
              <EmptyState
                title="جستجو انجام نشد."
                note="بک‌اند را بررسی کنید و دوباره تلاش کنید."
                action={
                  <button
                    className="filter-chip"
                    type="button"
                    onClick={() => runSearch(search.query)}
                  >
                    تلاش دوباره
                  </button>
                }
              >
                {search.error}
              </EmptyState>
            ) : search.status === "done" && !search.results.length ? (
              <EmptyState title="عکسی با این توضیح پیدا نشد.">
                یک عبارت دیگر را امتحان کنید؛ مثلاً مکان، رنگ، حس‌وحال یا چیزی که در عکس بود.
              </EmptyState>
            ) : galleryNotice ? (
              galleryNotice
            ) : (
              <Gallery
                items={displayed}
                feed={isFeed}
                onOpen={(index) => setLightboxIndex(index)}
              />
            )}
          </>
        )}

        {page === "library" && (
          <div className="compact-tools">
            <div className="filters" aria-label="فیلتر کتابخانه">
              {["All", ...CATEGORIES].map((category) => (
                <button
                  key={category}
                  className={`filter-chip${activeFilter === category ? " active" : ""}`}
                  type="button"
                  onClick={() => chooseFilter(category)}
                >
                  {category === "All" ? "همه" : category}
                </button>
              ))}
            </div>
            <div className="collection-count">
              {activeFilter === "All"
                ? `${fa(allPhotos.length)} عکس`
                : semantic.term === activeFilter && semantic.status === "done"
                  ? `${fa(semantic.results.length)} عکس نزدیک به «${activeFilter}»`
                  : `در حال بررسی «${activeFilter}»…`}
            </div>

            {activeFilter === "All" ? (
              galleryNotice || (
                <>
                  <Gallery
                    items={displayed}
                    onOpen={(index) => setLightboxIndex(index)}
                  />
                  {visibleCount < allPhotos.length && (
                    <button
                      className="load-more"
                      type="button"
                      onClick={() => setVisibleCount((count) => count + PAGE_SIZE)}
                    >
                      نمایش عکس‌های بیشتر
                    </button>
                  )}
                </>
              )
            ) : semanticBusy || semantic.status === "idle" ? (
              <LoadingPanel
                title={indexReady ? "در حال جستجو در گالری…" : "مدل در حال بررسی گالری…"}
                message={SEARCH_STAGES[stage]}
                note={modelNote}
              />
            ) : semantic.status === "error" ? (
              <EmptyState
                title="بررسی این مجموعه ممکن نشد."
                action={
                  <button
                    className="filter-chip"
                    type="button"
                    onClick={() => loadSemantic(activeFilter)}
                  >
                    تلاش دوباره
                  </button>
                }
              >
                {semantic.error}
              </EmptyState>
            ) : displayed.length ? (
              <Gallery items={displayed} onOpen={(index) => setLightboxIndex(index)} />
            ) : (
              <EmptyState title="عکسی در این مجموعه پیدا نشد.">
                عبارت دیگری از دسته‌بندی‌ها را امتحان کنید.
              </EmptyState>
            )}
          </div>
        )}

        {page === "albums" && (
          <>
            <div className="collection-count">{fa(1)} آلبوم</div>
            <div className="album-grid">
              <article
                className="album-card"
                role="button"
                tabIndex={0}
                onClick={() => goTo("library")}
                onKeyDown={(event) => {
                  if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault();
                    goTo("library");
                  }
                }}
              >
                <div className="album-cover">
                  {allPhotos.length ? (
                    allPhotos.slice(0, 3).map((photo) => (
                      <img key={photo.id} src={photo.image_url} alt="" loading="lazy" />
                    ))
                  ) : (
                    <span className="album-cover-empty">
                      <AlbumIcon />
                    </span>
                  )}
                </div>
                <div className="album-info">
                  <h3>همه‌ی عکس‌ها</h3>
                  <p>{fa(allPhotos.length)} عکس</p>
                </div>
              </article>
            </div>
            <div className="empty" style={{ marginTop: 19 }} role="status">
              <strong>آلبوم سفارشی هنوز ساخته نشده است.</strong>
              <p>
                این نسخه همه‌ی عکس‌ها را در یک آرشیو واحد نگه می‌دارد؛ برای دیدن همه‌ی تصاویر
                روی کارت بالا بزنید.
              </p>
            </div>
          </>
        )}

        {page === "categories" &&
          (selectedCategory === "All" ? (
            <div className="category-grid">
              {CATEGORIES.map((category, index) => {
                const cover = allPhotos.length
                  ? allPhotos[(index * 37 + 11) % allPhotos.length]
                  : null;
                return (
                  <article
                    className="category-card"
                    role="button"
                    tabIndex={0}
                    key={category}
                    onClick={() => openCategory(category)}
                    onKeyDown={(event) => {
                      if (event.key === "Enter" || event.key === " ") {
                        event.preventDefault();
                        openCategory(category);
                      }
                    }}
                  >
                    {cover ? (
                      <img src={cover.image_url} alt="" loading="lazy" />
                    ) : (
                      <span className="category-cover-fallback" aria-hidden="true" />
                    )}
                    <div className="category-info">
                      <h3>{category}</h3>
                      <p>جستجوی معنایی در آرشیو</p>
                    </div>
                  </article>
                );
              })}
            </div>
          ) : (
            <>
              <div className="section-head compact-section">
                <button
                  className="filter-chip"
                  type="button"
                  onClick={() => {
                    semanticRequest.current += 1;
                    setSelectedCategory("All");
                    setSemantic({ term: "", status: "idle", results: [], error: "" });
                    setLightboxIndex(null);
                  }}
                >
                  → همه‌ی دسته‌بندی‌ها
                </button>
                <span className="collection-count">{selectedCategory}</span>
              </div>

              {semanticBusy || semantic.status === "idle" ? (
                <LoadingPanel
                  title={indexReady ? "در حال جستجو در گالری…" : "مدل در حال بررسی گالری…"}
                  message={SEARCH_STAGES[stage]}
                  note={modelNote}
                />
              ) : semantic.status === "error" ? (
                <EmptyState
                  title="بررسی این دسته ممکن نشد."
                  action={
                    <button
                      className="filter-chip"
                      type="button"
                      onClick={() => loadSemantic(selectedCategory)}
                    >
                      تلاش دوباره
                    </button>
                  }
                >
                  {semantic.error}
                </EmptyState>
              ) : displayed.length ? (
                <Gallery items={displayed} onOpen={(index) => setLightboxIndex(index)} />
              ) : (
                <EmptyState title="عکسی در این دسته پیدا نشد.">
                  دسته‌ی دیگری را امتحان کنید.
                </EmptyState>
              )}
            </>
          ))}
      </main>

      {currentPhoto && (
        <div
          className="lightbox"
          ref={lightbox}
          role="dialog"
          aria-modal="true"
          aria-label={`نمایش ${currentPhoto.title}`}
          onClick={(event) => {
            if (event.target === event.currentTarget || event.target.tagName === "FIGURE") {
              setLightboxIndex(null);
            }
          }}
        >
          <div className="lb-bar">
            <button
              className="lb-btn"
              type="button"
              aria-label="بستن"
              title="بستن"
              onClick={() => setLightboxIndex(null)}
            >
              ✕
            </button>
            <span className="lb-count">
              {fa(lightboxIndex + 1)} / {fa(displayed.length)}
            </span>
            <button
              className="lb-btn"
              type="button"
              aria-label="تمام‌صفحه"
              title="تمام‌صفحه"
              onClick={() => {
                if (document.fullscreenElement) {
                  document.exitFullscreen().catch(() => {});
                  return;
                }
                const request = lightbox.current?.requestFullscreen;
                if (request) request.call(lightbox.current).catch(() => {});
              }}
            >
              <svg viewBox="0 0 24 24" aria-hidden="true">
                <path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5" />
              </svg>
            </button>
          </div>

          {displayed.length > 1 && (
            <>
              <button
                className="lb-nav lb-prev"
                type="button"
                aria-label="عکس قبلی"
                onClick={() =>
                  setLightboxIndex(
                    (index) => (index - 1 + displayed.length) % displayed.length,
                  )
                }
              >
                ›
              </button>
              <button
                className="lb-nav lb-next"
                type="button"
                aria-label="عکس بعدی"
                onClick={() => setLightboxIndex((index) => (index + 1) % displayed.length)}
              >
                ‹
              </button>
            </>
          )}

          <figure className="lb-stage">
            <img
              src={currentPhoto.image_url}
              alt={currentPhoto.title}
              onDoubleClick={() => {
                if (document.fullscreenElement) {
                  document.exitFullscreen().catch(() => {});
                  return;
                }
                const request = lightbox.current?.requestFullscreen;
                if (request) request.call(lightbox.current).catch(() => {});
              }}
            />
            <figcaption>
              {currentPhoto.title} · {currentPhoto.place}
            </figcaption>
          </figure>
        </div>
      )}

      <div className={`toast${toastMessage ? " show" : ""}`} role="status">
        {toastMessage}
      </div>
    </div>
  );
}

export default App;
