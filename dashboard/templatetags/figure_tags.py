from django import template

register = template.Library()

@register.filter
def get_figure_by_description(figures, description):
    """
    Given a queryset of Figure objects, return the figure whose description
    matches the given description, or None if not found.
    """
    return figures.filter(description=description).first()
