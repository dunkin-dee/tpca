; Class definitions
(class_definition
  name: (identifier) @class.name
  superclasses: (argument_list)? @class.bases
  body: (block) @class.body) @class.def

; Function / method definitions
(function_definition
  name: (identifier) @function.name
  parameters: (parameters) @function.params
  return_type: (type)? @function.return_type
  body: (block) @function.body) @function.def

; Function calls
(call function: (identifier) @call.simple)
(call function: (attribute
  object: (identifier) @call.object
  attribute: (identifier) @call.method))

; Imports
(import_statement name: (dotted_name) @import.module)
(import_from_statement
  module_name: (dotted_name) @import.from
  name: (dotted_name) @import.symbol)

; Docstrings
(function_definition body: (block .
  (expression_statement (string) @function.docstring)))
(class_definition body: (block .
  (expression_statement (string) @class.docstring)))

; Decorated definitions
(decorated_definition
  (decorator (identifier) @decorator.name)
  definition: (function_definition
    name: (identifier) @decorated.function))
