from pettingzoo.test import api_test

from aintelope.environments import savanna as sut

ENV = sut.env()


def test_pettingzoo_api():
    api_test(ENV)
