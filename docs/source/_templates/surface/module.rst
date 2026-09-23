{{ fullname | escape | underline }}

.. automodule:: {{ fullname }}
   :no-members:
   :no-index:

{% block modules -%}
{% if all_modules %}
Modules
-------

.. autosummary::
   :toctree:
   :recursive:
   :template: surface/module.rst
{% for item in all_modules %}
   {{ item }}
{%- endfor %}
{% endif %}
{%- endblock %}

{% if not all_modules %}
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

{% block functions -%}
{% if functions %}
Functions
---------

.. autosummary::
   :toctree:
   :template: engine/function.rst
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
   :template: engine/exception.rst
{% for item in exceptions %}
   {{ item }}
{%- endfor %}
{% endif %}
{%- endblock %}
{% endif %}
