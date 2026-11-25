import { useState } from "react";

function App() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [message, setMessage] = useState("Search your gallery by description.");
  const [loading, setLoading] = useState(false);

  async function search(event) {
    event.preventDefault();
    const text = query.trim();
    if (!text) {
      setMessage("Enter a description to search.");
      return;
    }

    setLoading(true);
    setMessage("");
    setResults([]);
    try {
      const response = await fetch("/api/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: text, limit: 12 }),
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || `Search failed (${response.status}).`);
      }
      setResults(data.results);
      setMessage(data.results.length ? "" : "No matching photos found.");
    } catch (error) {
      setMessage(
        error instanceof Error ? error.message : "The search request failed.",
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="app">
      <h1>Semantic Photo Search</h1>
      <form onSubmit={search}>
        <label htmlFor="query">Describe a photo</label>
        <div className="search-row">
          <input
            id="query"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="For example, a beach at sunset"
            maxLength={1000}
          />
          <button type="submit" disabled={loading}>
            {loading ? "Searching..." : "Search"}
          </button>
        </div>
      </form>

      {message && <p role="status">{message}</p>}

      <section className="results" aria-label="Search results">
        {results.map((result) => (
          <figure key={result.filename}>
            <img src={result.image_url} alt={result.filename} loading="lazy" />
            <figcaption>
              {result.filename} <span>{result.score.toFixed(3)}</span>
            </figcaption>
          </figure>
        ))}
      </section>
    </main>
  );
}

export default App;
