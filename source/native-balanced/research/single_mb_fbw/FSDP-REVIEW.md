# FSDPTurbo development-guide self-check

Scope: research branch planner-name admission and matching configuration docs.
Read the repository's complete `.agent/skills/fsdp-turbo-dev-guide/SKILL.md`.

|Rule|Result|
|---|---|
|1 device abstraction|No device calls added|
|2 torch_npu isolation|No imports added|
|3 op registration|No operators added|
|4 structure design|Extends existing immutable planner config; native compiler retains resource checks|
|5 small public surface|No extra runtime interface|
|6 module independence|No cross-module imports added|
|7 model dependency|No model imports|
|8 model adaptation|No modeling patches|
|9 model directory|No model files/configs added|
|10 config validation|Existing __post_init__ checks names only; config remains frozen|
|11 logging|No logging added|
|12 environment|No environment variables added|
|13 names|Explicit eplb_budget_N matches native policy names|
|14 minimality|Two code lines changed, no unrelated refactoring|
|15 reuse|Uses existing config validation boundary|
|16 comments|No non-English code comments added|
|17 API types|planner remains typed str; runtime legal-value checks documented|
|18 docs|Feature doc, user guide and menu updated|
|19 typography|New Chinese docs have no inter-language spaces|
|20 device terms|No abbreviated device names added|
|21 headings|No numbered feature heading|
|22 fresh review before PR|No PR opened; required before any future PR|
|23 safety|No serialization or command execution changes|

Device qualification and numerical correctness are separate from this style and
configuration review. See evidence and STATUS.json for actual run results.
