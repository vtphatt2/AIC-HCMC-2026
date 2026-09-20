// A short-lived display preview; callers must still fetch the current response.
export class RecentContextCache<T> {
  private entries = new Map<string, { json: string; bytes: number; expires: number }>();
  private bytes = 0;

  constructor(
    private maxEntries = 32,
    private maxBytes = 2 * 1024 * 1024,
    private maxAgeMs = 120_000,
  ) {}

  get(key: string): T | undefined {
    const entry = this.entries.get(key);
    if (!entry) return undefined;
    if (entry.expires <= Date.now()) {
      this.delete(key);
      return undefined;
    }
    this.entries.delete(key);
    this.entries.set(key, entry);
    // Isolate cached data from subsequent changes made by a consumer.
    return JSON.parse(entry.json);
  }

  set(key: string, value: T): void {
    const json = JSON.stringify(value);
    const bytes = 2 * (key.length + json.length);
    this.delete(key);
    if (this.maxEntries <= 0 || bytes > this.maxBytes) return;
    while (this.entries.size >= this.maxEntries || this.bytes + bytes > this.maxBytes) {
      this.delete(this.entries.keys().next().value!);
    }
    this.entries.set(key, { json, bytes, expires: Date.now() + this.maxAgeMs });
    this.bytes += bytes;
  }

  delete(key: string): void {
    const entry = this.entries.get(key);
    if (entry) this.bytes -= entry.bytes;
    this.entries.delete(key);
  }
}
