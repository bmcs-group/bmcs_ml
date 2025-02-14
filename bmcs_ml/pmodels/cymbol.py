import sympy as sp
import re

class Cymbol(sp.Symbol):
    '''Add a codename attribute to the sympy symbol to
    enable
    1) latex names with upper index or lower index
    2) transformation of symbol arrays to function attributes
    (e.g. state variable arrays passed as dictionaries to
     the corrector-predictor functions)
    3) generate lambdified function that can be inspected using
    import inspect
    inspect.getsource(lambdified_function)
    '''
    def __new__(cls, name, codename=None, **assumptions):
        obj = super().__new__(cls, name, **assumptions)
        obj.codename = codename or name
        return obj

    @property
    def T(self):
        """Pass through the transpose operator to allow also scalar variables in places 
        where vectors or matrices are expected"""
        return self

    def __getstate__(self):
        """Return state for pickling, including custom attributes."""
        # Use a tuple or dict that captures only non-standard data
        return self.codename

    def __setstate__(self, state):
        """Restore state from pickling."""
        # Set the codename directly from state
        self.codename = state

def _print_Symbol(self, expr):
    CodePrinter = sp.printing.codeprinter.CodePrinter
    if hasattr(expr, 'codename'):
        name = expr.codename
    else:
        name = super(CodePrinter, self)._print_Symbol(expr)
    return re.sub(r'[\\\{\}]', '', name)

sp.printing.codeprinter.CodePrinter._print_Symbol = _print_Symbol

from sympy.utilities.codegen import codegen

def ccode(cfun_name, sp_expr, cfile):
    '''Generate c function cfun_name for expr and directive name cfile
    '''
    return codegen((cfun_name, sp_expr), 'C89', cfile + '_' + cfun_name)

def cymbols(names, codenames=None, **kwargs):
    """
    Convenience function to create multiple Cymbol instances.

    Parameters
    ----------
    names : str
        Whitespace or comma separated list of variable names to be rendered using LaTeX.
    codenames : str, optional
        Whitespace or comma separated list of code names to be passed to the Cymbol constructor.
        If not provided, the names will be used as code names.

    Returns
    -------
    tuple
        Tuple of Cymbol instances.
    """
    var_list = [var.strip() for var in names.replace(',', ' ').split()]
    if codenames:
        code_list = [code.strip() for code in codenames.replace(',', ' ').split()]
    else:
        code_list = var_list
    return tuple(Cymbol(var, codename=code, **kwargs) for var, code in zip(var_list, code_list))
