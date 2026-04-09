/** Format an ISO date string into a localized "ru-RU" medium date with short time. */
export function formatDateTime(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return new Intl.DateTimeFormat("ru-RU", {
    dateStyle: "medium",
    timeStyle: "short"
  }).format(date);
}

/** Format a numeric dollar amount with a leading "$" sign and fixed decimal digits. */
export function formatMoney(value: number, digits = 4): string {
  return `$${value.toFixed(digits)}`;
}

/** Format a number as a percentage string with one decimal place. */
export function formatPercent(value: number): string {
  return `${value.toFixed(1)}%`;
}

/** Return the correct Slavic plural form (one/few/many) for the given count. */
export function pluralize(count: number, one: string, few: string, many: string): string {
  const mod10 = count % 10;
  const mod100 = count % 100;

  if (mod10 === 1 && mod100 !== 11) {
    return one;
  }
  if (mod10 >= 2 && mod10 <= 4 && (mod100 < 10 || mod100 >= 20)) {
    return few;
  }
  return many;
}
