#!/usr/bin/env bash

git checkout gh-pages
rm -rf _sources _static _modules
git checkout master docs/source docs/Makefile
git reset HEAD
cd docs
make clean && make html
cd ..
cp -r docs/build/html/* ./
rm -rf docs/source
rm docs/Makefile
git add -A
git commit -m "Update gh-pages for `git log master -1 --pretty=short --abbrev-commit`"
git push origin gh-pages
