{{ objname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

{% block attributes -%}
{% set own_attributes = [] %}
{% for item in all_attributes if not item.startswith("__") and item != "_abc_impl" and (not item.startswith("_") or item not in inherited_members) %}
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
{% for item in all_methods if not item.startswith("__") and (not item.startswith("_") or item not in inherited_members) %}
{% set _ = own_methods.append(item) %}
{% endfor %}
{% if own_methods %}
Methods
-------

.. autosummary::
   :toctree:
   :template: surface/method.rst
{% for item in own_methods %}
   ~{{ objname }}.{{ item }}
{%- endfor %}
{% endif %}
{%- endblock %}
