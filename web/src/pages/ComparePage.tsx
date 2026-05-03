import { Link } from "react-router-dom";
import { getResultById } from "@/api/backend";
import { useAppPreferences } from "@/context/AppPreferences";
import { ResultCard } from "@/components/ResultCard";

export function ComparePage() {
  const { compareSlots, clearCompare } = useAppPreferences();
  const [a, b] = compareSlots;
  const ra = a ? getResultById(a) : undefined;
  const rb = b ? getResultById(b) : undefined;

  const missingMeta = (Boolean(a) && !ra) || (Boolean(b) && !rb);
  const hasAnySlot = Boolean(a ?? b);

  return (
    <div className="page compare-page">
      <header className="page-header">
        <h1>Paired comparison</h1>
        <p className="muted">
          Select two query results for side-by-side inspection. Slots are filled from the{" "}
          <strong>Compare slot</strong> action on each result card.
        </p>
      </header>
      {missingMeta ? (
        <aside className="panel-alert panel-alert--soft" role="status">
          One or both recordings could not be found. Run a fresh search and re-add them to compare.
        </aside>
      ) : null}
      {!hasAnySlot ? (
        <section className="panel">
          <div className="empty-panel-hint muted">
            <p className="panel__flush">No recordings in the comparison queue.</p>
            <p className="small">
              <Link to="/query">Return to Query</Link> and assign two clips with <strong>Compare slot</strong>.
            </p>
          </div>
        </section>
      ) : (
        <>
          <div className="compare-toolbar">
            <button type="button" className="btn btn--outline" onClick={clearCompare}>
              Clear slots
            </button>
          </div>
          <div className="compare-grid">
            <div>
              {!a ? (
                <p className="compare-slot muted">Slot A · unassigned</p>
              ) : !ra ? (
                <p className="compare-slot muted panel-alert panel-alert--error">
                  Slot A · recording not found. <Link to="/query">Run a search</Link> and re-add it.
                </p>
              ) : (
                <ResultCard result={ra} />
              )}
            </div>
            <div>
              {!b ? (
                <p className="compare-slot muted">Slot B · select a second recording from Query</p>
              ) : !rb ? (
                <p className="compare-slot muted panel-alert panel-alert--error">
                  Slot B · recording not found. <Link to="/query">Run a search</Link> and re-add it.
                </p>
              ) : (
                <ResultCard result={rb} />
              )}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
