; ── JavaScript Tree-sitter Query ────────────────────────────────────────────
; Used by ASTIndexer when language == 'javascript'
; Node names follow tree-sitter-javascript grammar.

; ── Class declarations ───────────────────────────────────────────────────────
(class_declaration
  name: (identifier) @class.name
  (class_heritage
    (identifier) @class.superclass)?
  body: (class_body) @class.body)

; Class expression assigned to a variable
(variable_declarator
  name: (identifier) @class.name
  value: (class
    (class_heritage
      (identifier) @class.superclass)?
    body: (class_body) @class.body))

; ── Method definitions ───────────────────────────────────────────────────────
(method_definition
  name: (property_identifier) @method.name
  parameters: (formal_parameters) @method.params
  body: (statement_block) @method.body)

; Static method definitions
(method_definition
  "static"
  name: (property_identifier) @method.name
  parameters: (formal_parameters) @method.params
  body: (statement_block) @method.body)

; ── Function declarations ────────────────────────────────────────────────────
(function_declaration
  name: (identifier) @function.name
  parameters: (formal_parameters) @function.params
  body: (statement_block) @function.body)

; Named function expressions assigned to variables (const/let/var)
(lexical_declaration
  (variable_declarator
    name: (identifier) @function.name
    value: (function
      parameters: (formal_parameters) @function.params
      body: (statement_block) @function.body)))

; Arrow functions assigned to variables
(lexical_declaration
  (variable_declarator
    name: (identifier) @function.name
    value: (arrow_function
      parameters: (formal_parameters) @function.params
      body: (_) @function.body)))

; Exported function declarations
(export_statement
  declaration: (function_declaration
    name: (identifier) @function.name
    parameters: (formal_parameters) @function.params
    body: (statement_block) @function.body))

; Exported class declarations
(export_statement
  declaration: (class_declaration
    name: (identifier) @class.name
    (class_heritage
      (identifier) @class.superclass)?
    body: (class_body) @class.body))

; ── Import statements ─────────────────────────────────────────────────────────
(import_statement
  source: (string) @import.source)

(import_statement
  (import_clause
    (named_imports
      (import_specifier
        name: (identifier) @import.symbol))))

(import_statement
  (import_clause
    (identifier) @import.default))

; ── JSDoc / leading comments (captured as docstrings) ────────────────────────
; Tree-sitter exposes comments as nodes; the indexer filters for /** … */ style.
(comment) @docstring.candidate

; ── Call expressions ──────────────────────────────────────────────────────────
(call_expression
  function: (identifier) @call.simple)

(call_expression
  function: (member_expression
    object: (identifier) @call.object
    property: (property_identifier) @call.method))

; new expressions (constructor calls)
(new_expression
  constructor: (identifier) @call.constructor)
