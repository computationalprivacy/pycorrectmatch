# CorrectMatch
# Copyright © 2019 Université catholique de Louvain, UCLouvain
# Copyright © 2019 Imperial College London
# by Luc Rocher, Julien Hendrickx, Yves-Alexandre de Montjoye
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from setuptools import setup, find_packages

setup(
    name = 'CorrectMatch',
    version = '0.1.0',
    url = 'https://github.com/computationalprivacy/pycorrectmatch',
    author = 'Luc Rocher',
    author_email = 'luc@rocher.lc',
    description = 'Python wrapper for the Julia package CorrectMatch.jl',
    packages = find_packages(),    
    install_requires = ['numpy >= 1.15.0', 'julia >= 0.1.5'],
)