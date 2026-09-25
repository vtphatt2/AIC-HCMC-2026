import type { SubmissionQueryType, SubmissionRow } from '../../types';

export function submissionUnit(videoId: string): 'frames' | 'milliseconds' {
  return videoId.startsWith('N') ? 'milliseconds' : 'frames';
}

export function parsePosition(raw: string): number {
  if (!/^\d+$/.test(raw.trim())) throw new Error(`Invalid position "${raw}"`);
  const value = Number(raw.trim());
  if (!Number.isSafeInteger(value)) throw new Error('Position exceeds safe integer range');
  return value;
}

export function parseCsv(text: string): string[][] {
  const records: string[][] = [];
  let row: string[] = [], field = '', quoted = false, closed = false;
  for (let i = 0; i < text.length; i++) {
    const char = text[i];
    if (quoted) {
      if (char === '"' && text[i+1] === '"') { field += '"'; i++; }
      else if (char === '"') { quoted = false; closed = true; }
      else field += char;
    } else if (char === '"') {
      if (field || closed) throw new Error('Unexpected CSV quote');
      quoted = true;
    } else if (char === ',' || char === '\n' || char === '\r') {
      row.push(field); field = ''; closed = false;
      if (char !== ',') {
        if (char === '\r' && text[i+1] === '\n') i++;
        if (row.some(value => value !== '')) records.push(row);
        row = [];
      }
    } else {
      if (closed) throw new Error('Unexpected text after CSV quote');
      field += char;
    }
  }
  if (quoted) throw new Error('Unclosed CSV quote');
  if (field || row.length || closed) { row.push(field); records.push(row); }
  return records;
}

function quote(value: string): string {
  return /[",\r\n]/.test(value) ? `"${value.replace(/"/g, '""')}"` : value;
}

export function validateSubmissionRow(type: SubmissionQueryType, row: SubmissionRow, exporting = false): void {
  if (!/^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/.test(row.videoId)) throw new Error('Invalid video ID');
  if (!row.frames.length || row.frames.some(n => !Number.isSafeInteger(n) || n < 0)) throw new Error('Invalid positions');
  if (type !== 'trake' && row.frames.length !== 1) throw new Error('KIS/QA require one position');
  if (type === 'trake' && row.videoId.startsWith('N')) throw new Error('N videos cannot be submitted in TRAKE');
  if ((row.answer?.length ?? 0) > 100) throw new Error('Answer must be 100 characters or fewer');
  if (exporting && row.videoId.startsWith('N') && (row.unit !== 'milliseconds' || row.timingStatus !== 'verified')) {
    throw new Error('N row needs verified source timing before export');
  }
}

export function serializeSubmissionRow(type: SubmissionQueryType, row: SubmissionRow): string {
  validateSubmissionRow(type, row);
  return [row.videoId, ...row.frames.map(String), ...(type === 'qa' ? [row.answer ?? ''] : [])].map(quote).join(',');
}

export function parseSubmissionCsv(type: SubmissionQueryType, text: string): SubmissionRow[] {
  return parseCsv(text).map((fields, index) => {
    if ((type === 'kis' && fields.length !== 2) || (type === 'qa' && fields.length !== 3) ||
        (type === 'trake' && fields.length < 2)) throw new Error(`Row ${index+1}: incorrect CSV column count`);
    const row: SubmissionRow = { videoId: fields[0], frames: fields.slice(1, type === 'qa' ? 2 : undefined).map(parsePosition),
      unit: submissionUnit(fields[0]), ...(type === 'qa' ? { answer: fields[2] } : {}) };
    validateSubmissionRow(type, row);
    return row;
  });
}
