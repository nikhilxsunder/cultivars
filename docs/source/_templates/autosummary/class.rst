{{ objname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

{% block attributes -%}
{% if attributes %}
Attributes
----------

.. autosummary::
   :toctree:
{% for item in attributes %}
   ~{{ objname }}.{{ item }}
{%- endfor %}
{% endif %}
{%- endblock %}

{% block methods -%}
{% set public_methods = methods | reject("equalto", "__init__") | list %}
{% if public_methods %}
Methods
-------

.. autosummary::
   :toctree:
{% for item in public_methods %}
   ~{{ objname }}.{{ item }}
{%- endfor %}
{% endif %}
{%- endblock %}
