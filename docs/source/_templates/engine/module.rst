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

{% block attributes -%}
{% if all_attributes %}
Type aliases and constants
--------------------------

.. autosummary::
   :toctree:
   :template: engine/data.rst
{% for item in all_attributes %}
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
