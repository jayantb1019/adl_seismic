python adl_tests.py -dataset faciesmark -cnt truly_random -cnl 0.01 -tnt gaussian -tnl 0.001
python adl_tests.py -dataset faciesmark -cnt truly_random -cnl 0.01 -tnt gaussian -tnl 0.01
python adl_tests.py -dataset faciesmark -cnt truly_random -cnl 0.01 -tnt gaussian -tnl 0.025
python adl_tests.py -dataset faciesmark -cnt truly_random -cnl 0.01 -tnt gaussian -tnl 0.05
python adl_tests.py -dataset faciesmark -cnt truly_random -cnl 0.01 -tnt gaussian -tnl 0.1
python adl_tests.py -dataset faciesmark -cnt truly_random -cnl 0.01 -tnt gaussian -tnl 0.25
python adl_tests.py -dataset faciesmark -cnt truly_random -cnl 0.01 -tnt gaussian -tnl 0.5
python adl_tests.py -dataset faciesmark -cnt truly_random -cnl 0.01 -tnt poisson -tnl 0.01
python adl_tests.py -dataset faciesmark -cnt truly_random -cnl 0.01 -tnt mixed -tnl 0.01
python adl_tests.py -dataset faciesmark -cnt truly_random -cnl 0.01 -tnt lpf -tnl 0.01
python adl_tests.py -dataset stdata12 -cnt truly_random -cnl 0.01 -tnt gaussian -tnl 0.001
python adl_tests.py -dataset stdata12 -cnt truly_random -cnl 0.01 -tnt gaussian -tnl 0.01
python adl_tests.py -dataset stdata12 -cnt truly_random -cnl 0.01 -tnt gaussian -tnl 0.025
python adl_tests.py -dataset stdata12 -cnt truly_random -cnl 0.01 -tnt gaussian -tnl 0.05
python adl_tests.py -dataset stdata12 -cnt truly_random -cnl 0.01 -tnt gaussian -tnl 0.1
python adl_tests.py -dataset stdata12 -cnt truly_random -cnl 0.01 -tnt gaussian -tnl 0.25
python adl_tests.py -dataset stdata12 -cnt truly_random -cnl 0.01 -tnt gaussian -tnl 0.5
python adl_tests.py -dataset stdata12 -cnt truly_random -cnl 0.01 -tnt poisson -tnl 0.01
python adl_tests.py -dataset stdata12 -cnt truly_random -cnl 0.01 -tnt mixed -tnl 0.01
python adl_tests.py -dataset stdata12 -cnt truly_random -cnl 0.01 -tnt lpf -tnl 0.01
python adl_tests.py -dataset interpretation -cnt truly_random -cnl 0.01 -tnt gaussian -tnl 0.001
python adl_tests.py -dataset interpretation -cnt truly_random -cnl 0.01 -tnt gaussian -tnl 0.01
python adl_tests.py -dataset interpretation -cnt truly_random -cnl 0.01 -tnt gaussian -tnl 0.025
python adl_tests.py -dataset interpretation -cnt truly_random -cnl 0.01 -tnt gaussian -tnl 0.05
python adl_tests.py -dataset interpretation -cnt truly_random -cnl 0.01 -tnt gaussian -tnl 0.1
python adl_tests.py -dataset interpretation -cnt truly_random -cnl 0.01 -tnt gaussian -tnl 0.25
python adl_tests.py -dataset interpretation -cnt truly_random -cnl 0.01 -tnt gaussian -tnl 0.5
python adl_tests.py -dataset interpretation -cnt truly_random -cnl 0.01 -tnt poisson -tnl 0.01
python adl_tests.py -dataset interpretation -cnt truly_random -cnl 0.01 -tnt mixed -tnl 0.01
python adl_tests.py -dataset interpretation -cnt truly_random -cnl 0.01 -tnt lpf -tnl 0.01
cd ..
git add .
git commit -m 'truly random tests completed'
git push