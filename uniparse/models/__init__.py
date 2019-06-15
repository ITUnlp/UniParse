"""Models module containing reimplementations of state-of-the-art parsers."""

from uniparse.models.dynet.kiperwasser import Kiperwasser as dykiperwasser
from uniparse.models.dynet.varab import Varab as dyvarab
from uniparse.models.dynet.dozat import Dozat as dydozat
from uniparse.models.pytorch_models.kiperwasser import Kiperwasser as ptkiperwasser

INCLUDED_PARSERS = {
    "dynet-kiperwasser": dykiperwasser,
    "dynet-dozat": dydozat,
    "varab": dyvarab,
    "pytorch-kiperwasser": ptkiperwasser
}