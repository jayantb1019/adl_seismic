python adl_tests.py -dataset faciesmark -cnt truly_random -cnl 0.01 -tnt truly_random -tnl 0.01
python adl_tests.py -dataset interpretation -cnt truly_random -cnl 0.01 -tnt truly_random -tnl 0.01
python adl_tests.py -dataset stdata12 -cnt truly_random -cnl 0.01 -tnt truly_random -tnl 0.01
cd ..
git add .
git commit -m 'truly random tests with truly random noise'
git push