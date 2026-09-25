{{ fullname | escape | underline }}

.. automodule:: {{ fullname }}
   :no-members:
   :no-index:

{% block modules -%}
{% if modules %}
Modules
-------

.. autosummary::
   :toctree:
   :recursive:
   :template: surface/module.rst
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
   :template: surface/class.rst
{% for item in classes %}
   {{ item }}
{%- endfor %}
{% endif %}
{%- endblock %}
