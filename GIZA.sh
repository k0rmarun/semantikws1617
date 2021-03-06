#!/bin/bash

plain2snt_binary="/home/niels/Downloads/mgiza/mgizapp/build/inst/bin/plain2snt"
mkcls_binary="/home/niels/Downloads/mgiza/mgizapp/build/inst/bin/mkcls"
snt2cooc_binary="/home/niels/Downloads/mgiza/mgizapp/build/inst/bin/snt2cooc"
giza_binary="/home/niels/Downloads/mgiza/mgizapp/build/inst/bin/mgiza"
tempdir="/home/niels/tmp/result/"

cd $tempdir

echo "Running plain2snt"
eval "$plain2snt_binary from to"

echo "Running mkcls"
eval "$mkcls_binary -pfrom -Vfrom.vcb.classes"
eval "$mkcls_binary -pto -Vto.vcb.classes"

echo "Running snt2cooc"
eval "$snt2cooc_binary from_to.cooc from.vcb to.vcb from_to.snt"

echo "Running giza"
cp from_to.snt fromto.snt #Crashes if _ in name
eval "$giza_binary -S from.vcb -T to.vcb -C fromto.snt -CoocurrenceFile from_to.cooc"

echo "Collecting result files"
cat *A3.final.part* >> result