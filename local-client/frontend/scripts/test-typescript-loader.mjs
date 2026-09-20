// Let Node 18+ run the TypeScript unit tests using the existing dev dependency.
import { readFile } from "node:fs/promises";
import ts from "typescript";

export async function load(url, context, nextLoad) {
  if (new URL(url).pathname.endsWith(".ts")) {
    return {
      format: "module",
      shortCircuit: true,
      source: ts.transpileModule(await readFile(new URL(url), "utf8"), {
        compilerOptions: {
          module: ts.ModuleKind.ESNext,
          target: ts.ScriptTarget.ES2020,
        },
      }).outputText,
    };
  }
  return nextLoad(url, context);
}
