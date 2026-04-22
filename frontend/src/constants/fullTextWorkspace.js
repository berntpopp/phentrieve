export const SIDEBAR_MODE_CASE = 'case';
export const SIDEBAR_MODE_INSPECTOR = 'inspector';

function freezeBand(label, min) {
  return Object.freeze({ label, min });
}

export const CONFIDENCE_BANDS = Object.freeze({
  high: freezeBand('High', 0.85),
  medium: freezeBand('Medium', 0.6),
  low: freezeBand('Low', 0.0),
});
