; ── TypeScript Tree-sitter Query ─────────────────────────────────────────────
; Used by ASTIndexer when language == 'typescript' or 'tsx'
; Inherits all JavaScript patterns; adds TS-specific constructs.
; Node names follow tree-sitter-typescript grammar.

; ── Class declarations (with optional type parameters) ───────────────────────
(class_declaration
  name: (type_identifier) @class.name
  (class_heritage
    (implements_clause
      (type_identifier) @class.implements)?)?
  (class_heritage
    (extends_clause
      value: (identifier) @class.superclass)?)?
  body: (class_body) @class.body)

(export_statement
  declaration: (class_declaration
    name: (type_identifier) @class.name
    body: (class_body) @class.body))

; ── Interface declarations ────────────────────────────────────────────────────
(interface_declaration
  name: (type_identifier) @interface.name
  body: (interface_body) @interface.body)

(export_statement
  declaration: (interface_declaration
    name: (type_identifier) @interface.name
    body: (interface_body) @interface.body))

; ── Type aliases ──────────────────────────────────────────────────────────────
(type_alias_declaration
  name: (type_identifier) @type.name
  value: (_) @type.value)

(export_statement
  declaration: (type_alias_declaration
    name: (type_identifier) @type.name
    value: (_) @type.value))

; ── Enum declarations ─────────────────────────────────────────────────────────
(enum_declaration
  name: (identifier) @enum.name
  body: (enum_body) @enum.body)

(export_statement
  declaration: (enum_declaration
    name: (identifier) @enum.name
    body: (enum_body) @enum.body))

; ── Method definitions (with return type annotations) ────────────────────────
(method_definition
  name: (property_identifier) @method.name
  parameters: (formal_parameters) @method.params
  return_type: (type_annotation)? @method.return_type
  body: (statement_block) @method.body)

; Abstract method signatures (no body)
(abstract_method_signature
  name: (property_identifier) @method.name
  parameters: (formal_parameters) @method.params
  return_type: (type_annotation)? @method.return_type)

; ── Function declarations (with return type annotations) ─────────────────────
(function_declaration
  name: (identifier) @function.name
  parameters: (formal_parameters) @function.params
  return_type: (type_annotation)? @function.return_type
  body: (statement_block) @function.body)

(export_statement
  declaration: (function_declaration
    name: (identifier) @function.name
    parameters: (formal_parameters) @function.params
    return_type: (type_annotation)? @function.return_type
    body: (statement_block) @function.body))

; Arrow functions with type annotations
(lexical_declaration
  (variable_declarator
    name: (identifier) @function.name
    type: (type_annotation)? @function.type_annotation
    value: (arrow_function
      parameters: (formal_parameters) @function.params
      return_type: (type_annotation)? @function.return_type
      body: (_) @function.body)))

; Exported arrow functions (e.g. export const Foo = () => {}, dominant Vite pattern)
(export_statement
  declaration: (lexical_declaration
    (variable_declarator
      name: (identifier) @function.name
      type: (type_annotation)? @function.type_annotation
      value: (arrow_function
        parameters: (formal_parameters) @function.params
        return_type: (type_annotation)? @function.return_type
        body: (_) @function.body))))

; ── Import / export statements ────────────────────────────────────────────────
(import_statement
  source: (string) @import.source)

(import_statement
  (import_clause
    (named_imports
      (import_specifier
        name: (identifier) @import.symbol))))

; Type-only imports (TS 3.8+)
(import_statement
  "type"
  source: (string) @import.type_source)

; ── Property signatures in interfaces ────────────────────────────────────────
(property_signature
  name: (property_identifier) @property.name
  type: (type_annotation)? @property.type)

; ── TSDoc / JSDoc comments ────────────────────────────────────────────────────
(comment) @docstring.candidate

; ── Call expressions ──────────────────────────────────────────────────────────
(call_expression
  function: (identifier) @call.simple)

(call_expression
  function: (member_expression
    object: (identifier) @call.object
    property: (property_identifier) @call.method))

(new_expression
  constructor: (identifier) @call.constructor)
