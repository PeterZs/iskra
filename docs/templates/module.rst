{{ fullname | escape | underline}}

.. automodule:: {{ fullname }}

{% if classes %}
.. autosummary::
   :toctree:
{% for item in classes %}
   {{ item }}
{%- endfor %}
{% endif %}

{% if functions %}
.. autosummary::
   :toctree:
{% for item in functions %}
   {{ item }}
{%- endfor %}
{% endif %}