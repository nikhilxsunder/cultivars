{{ objname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :show-inheritance:

{% block attributes -%}
{% set own_attributes = [] %}
{% for item in all_attributes if not item.startswith("__") %}
{% set _ = own_attributes.append(item) %}
{% endfor %}
{% if own_attributes %}
Attributes
----------

.. autosummary::
   :toctree:
   :template: engine/attribute.rst
{% for item in own_attributes %}
   ~{{ objname }}.{{ item }}
{%- endfor %}
{% endif %}
{%- endblock %}

{% block methods -%}
{% set own_methods = [] %}
{% for item in all_methods if not item.startswith("__") %}
{% set _ = own_methods.append(item) %}
{% endfor %}
{% if own_methods %}
Methods
-------

.. autosummary::
   :toctree:
   :template: engine/method.rst
{% for item in own_methods %}
   ~{{ objname }}.{{ item }}
{%- endfor %}
{% endif %}
{%- endblock %}
