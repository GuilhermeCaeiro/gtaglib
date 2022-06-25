from setuptools import find_packages, setup

setup(
    name="gtaglib",
    packages=find_packages(include=["gtaglib"]),
    version="0.1.0",
    description="GTAGLIB - A library to generate tags and tag clouds.",
    url="https://github.com/GuilhermeCaeiro/gtaglib",
    author="Guilherme Caeiro de Mattos",
    author_email="mattosgc@cos.ufrj.br",
    license="MIT",
    install_requires=[
    	"scikit-learn>=1.0.2", 
    	"pandas>=1.3.5",
    	"numpy>=1.21.6",
    	"wordcloud>=1.8.1",
    	"nltk>=3.7",
    	"unidecode>=1.3.4"
	],
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
    test_suite="tests",
)