#!/bin/bash

mkdir data && cd data

wget https://www.bcgsc.ca/downloads/111_reference_epigenomes/112epigenomes/5mC/SBS_Removed_E027_E064_Fixed_E012/header
wget https://www.bcgsc.ca/downloads/111_reference_epigenomes/112epigenomes/5mC/SBS_Removed_E027_E064_Fixed_E012/EG.mnemonics.name.xls

wget https://www.bcgsc.ca/downloads/111_reference_epigenomes/112epigenomes/5mC/SBS_Removed_E027_E064_Fixed_E012/FractionalMethylation.tar.gz -O FractionalMethylation.tar.gz && mkdir FractionalMethylation && tar -xvf FractionalMethylation.tar.gz -C FractionalMethylation --strip-components=2

wget https://www.bcgsc.ca/downloads/111_reference_epigenomes/112epigenomes/5mC/SBS_Removed_E027_E064_Fixed_E012/ReadCoverage.tar.gz -O ReadCoverage.tar.gz && mkdir ReadCoverage && tar -xvf ReadCoverage.tar.gz -C ReadCoverage --strip-components=1