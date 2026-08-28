export interface ParsedSessionNames {
  valid: string[];
  existing: string[];
  duplicates: string[];
  invalid: string[];
}

const SESSION_NAME_PATTERN = /^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/;

export function sessionNameError(name: string): string | null {
  const trimmed = name.trim();
  if (!trimmed) return "Enter a session name.";
  if (trimmed.length > 64) return "Session names can contain at most 64 characters.";
  if (!/^[A-Za-z0-9]/.test(trimmed)) return "Session names must start with a letter or number.";
  if (!SESSION_NAME_PATTERN.test(trimmed)) {
    return "Use letters, numbers, underscores, and hyphens only.";
  }
  return null;
}

export function parseSessionNames(
  input: string,
  existingNames: Iterable<string> = [],
): ParsedSessionNames {
  const result: ParsedSessionNames = {
    valid: [],
    existing: [],
    duplicates: [],
    invalid: [],
  };
  const existing = new Set(existingNames);
  const seen = new Set<string>();

  for (const rawLine of input.split(/\r?\n/)) {
    const name = rawLine.trim();
    if (!name) continue;
    if (seen.has(name)) {
      if (!result.duplicates.includes(name)) result.duplicates.push(name);
      continue;
    }
    seen.add(name);

    if (sessionNameError(name)) result.invalid.push(name);
    else if (existing.has(name)) result.existing.push(name);
    else result.valid.push(name);
  }

  return result;
}
