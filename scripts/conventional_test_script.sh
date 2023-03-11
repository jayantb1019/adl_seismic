python conventional_tests.py -dataset stdata12 -tnt poisson -tnl 0.05
python conventional_tests.py -dataset stdata12 -tnt gaussian -tnl 0.001
python conventional_tests.py -dataset stdata12 -tnt gaussian -tnl 0.01
python conventional_tests.py -dataset stdata12 -tnt gaussian -tnl 0.025
python conventional_tests.py -dataset stdata12 -tnt gaussian -tnl 0.05
python conventional_tests.py -dataset stdata12 -tnt gaussian -tnl 0.1
python conventional_tests.py -dataset stdata12 -tnt gaussian -tnl 0.25
python conventional_tests.py -dataset stdata12 -tnt gaussian -tnl 0.5
python conventional_tests.py -dataset stdata12 -tnt mixed -tnl 0.0
python conventional_tests.py -dataset stdata12 -tnt poisson -tnl 0.05
python conventional_tests.py -dataset faciesmark -tnt truly_random -tnl 0.0
python conventional_tests.py -dataset interpretation -tnt truly_random -tnl 0.0
python conventional_tests.py -dataset stdata12 -tnt truly_random -tnl 0.0
python conventional_tests.py -dataset interpretation -tnt gaussian -tnl 0.001
python conventional_tests.py -dataset interpretation -tnt gaussian -tnl 0.01
python conventional_tests.py -dataset interpretation -tnt gaussian -tnl 0.025
python conventional_tests.py -dataset interpretation -tnt gaussian -tnl 0.05
python conventional_tests.py -dataset interpretation -tnt gaussian -tnl 0.1
python conventional_tests.py -dataset interpretation -tnt gaussian -tnl 0.25
python conventional_tests.py -dataset interpretation -tnt gaussian -tnl 0.5
python conventional_tests.py -dataset interpretation -tnt mixed -tnl 0.0
python conventional_tests.py -dataset interpretation -tnt poisson -tnl 0.05
cd ..
git add .
git commit -m 'conventional tests completed'
