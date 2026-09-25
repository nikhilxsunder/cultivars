{{ fullname | escape | underline }}

.. automodule:: {{ fullname }}
   :no-members:

{% block classes -%}
{% if all_classes %}
Classes
-------

.. autosummary::
   :toctree:
   :nosignatures:
   :template: engine/class.rst
{% for item in all_classes %}
   {{ item }}
{%- endfor %}
{% endif %}
{%- endblock %}

{% block functions -%}
{% if all_functions %}
Functions
---------

.. autosummary::
   :toctree:
   :template: engine/function.rst
{% for item in all_functions %}
   {{ item }}
{%- endfor %}
{% endif %}
{%- endblock %}

{% block exceptions -%}
{% if all_exceptions %}
Exceptions
----------

.. autosummary::
   :toctree:
   :nosignatures:
   :template: engine/exception.rst
{% for item in all_exceptions %}
   {{ item }}
{%- endfor %}
{% endif %}
{%- endblock %}

{% block data -%}
{% set documented = (all_modules if all_modules is defined else []) + all_classes + all_functions + all_exceptions %}
{% set data = [] %}
{% for item in members if item not in documented and not item.startswith("__") %}
{% set _ = data.append(item) %}
{% endfor %}
{% if data %}
Constants and defaults
----------------------

.. autosummary::
   :toctree:
   :template: engine/data.rst
{% for item in data %}
   {{ item }}
{%- endfor %}
{% endif %}
{%- endblock %}
