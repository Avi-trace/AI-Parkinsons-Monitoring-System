// Turns `eslint -f json` output into GitHub Actions annotations so lint problems
// show inline on pull requests. Usage: eslint . -f json | node scripts/eslint-annotate.mjs
import { relative } from "node:path";

let input = "";
for await (const chunk of process.stdin) input += chunk;
const results = JSON.parse(input || "[]");
let errors = 0;
for (const file of results) {
  for (const m of file.messages) {
    const level = m.severity === 2 ? "error" : "warning";
    if (level === "error") errors += 1;
    const path = `frontend/frontend/${relative(process.cwd(), file.filePath)}`;
    const msg = `${m.message}${m.ruleId ? ` (${m.ruleId})` : ""}`.replace(/\r?\n/g, " ");
    console.log(`::${level} file=${path},line=${m.line ?? 1},col=${m.column ?? 1}::${msg}`);
  }
}
console.log(`${errors} lint error(s)`);
process.exit(errors ? 1 : 0);
