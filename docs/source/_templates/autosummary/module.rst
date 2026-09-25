{{ fullname | escape | underline }}

.. automodule:: {{ fullname }}
   :no-members:

{% block modules -%}
{% if modules %}
Modules
-------

.. autosummary::
   :toctree:
   :recursive:
{% for item in modules %}
   {{ item }}
{%- endfor %}
{% endif %}
{%- endblock %}

{% block classes -%}
{% if classes %}
Classes
-------

.. autosummary::
   :toctree:
   :nosignatures:
{% for item in classes %}
   {{ item }}
{%- endfor %}
{% endif %}
{%- endblock %}

{% block functions -%}
{% if functions %}
Functions
---------

.. autosummary::
   :toctree:
{% for item in functions %}
   {{ item }}
{%- endfor %}
{% endif %}
{%- endblock %}

{% block exceptions -%}
{% if exceptions %}
Exceptions
----------

.. autosummary::
   :toctree:
   :nosignatures:
{% for item in exceptions %}
   {{ item }}
{%- endfor %}
{% endif %}
{%- endblock %}

{% block data -%}
{% set documented = (modules if modules is defined else []) + classes + functions + exceptions %}
{% set data = [] %}
{% for item in members if item not in documented and not item.startswith("_") %}
{% set _ = data.append(item) %}
{% endfor %}
{% if data %}
Type aliases and constants
--------------------------

.. autosummary::
   :toctree:
   :template: autosummary/data.rst
{% for item in data %}
   {{ item }}
{%- endfor %}
{% endif %}
{%- endblock %}
