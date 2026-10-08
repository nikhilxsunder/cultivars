{{ objname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :no-index:
   :show-inheritance:

{% block attributes -%}
{% set own_private_attributes = [] %}
{% for item in all_attributes if item.startswith("_") and not item.startswith("__") and item != "_abc_impl" and item not in inherited_members %}
{% set _ = own_private_attributes.append(item) %}
{% endfor %}
{% if own_private_attributes %}
Private attributes
------------------

.. autosummary::
   :toctree:
   :template: surface/attribute.rst
{% for item in own_private_attributes %}
   ~{{ objname }}.{{ item }}
{%- endfor %}
{% endif %}
{%- endblock %}

{% block methods -%}
{% set own_private_methods = [] %}
{% for item in all_methods if item.startswith("_") and not item.startswith("__") and item not in inherited_members %}
{% set _ = own_private_methods.append(item) %}
{% endfor %}
{% if own_private_methods %}
Private methods
---------------

.. autosummary::
   :toctree:
   :template: surface/method.rst
{% for item in own_private_methods %}
   ~{{ objname }}.{{ item }}
{%- endfor %}
{% endif %}
{%- endblock %}
