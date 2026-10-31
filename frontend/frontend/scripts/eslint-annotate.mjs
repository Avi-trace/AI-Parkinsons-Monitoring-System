// Turns `eslint -f json` output into GitHub Actions annotations so lint problems
// show inline on pull requests. Usage: eslint . -f json | node scripts/eslint-annotate.mjs
import { relative } from "node:path";

let input = "";
for await (const chunk of process.stdin) input += chunk;
if (!input.trim()) {
  console.log("::error title=eslint::no lint report produced");
  process.exit(2);
}
const results = JSON.parse(input);
let errors = 0;
const all = [];
for (const file of results) {
  for (const m of file.messages) {
    const level = m.severity === 2 ? "error" : "warning";
    if (level === "error") errors += 1;
    const path = `frontend/frontend/${relative(process.cwd(), file.filePath)}`;
    const msg = `${m.message}${m.ruleId ? ` (${m.ruleId})` : ""}`.replace(/\r?\n/g, " ");
    console.log(`::${level} file=${path},line=${m.line ?? 1},col=${m.column ?? 1}::${msg}`);
    all.push(`${path.replace("frontend/frontend/", "")}:${m.line} ${m.ruleId || "?"}`);
  }
}
console.log(`${errors} lint error(s)`);
// GitHub only displays 10 annotations per level, so also emit one full summary.
if (all.length) console.log(`::notice title=eslint summary (${all.length})::${all.join(" | ")}`);
process.exit(errors ? 1 : 0);
