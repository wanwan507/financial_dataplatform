cd lib
rm -rf *
cd ..
rm main-built.js
babel --presets react,es2015 --watch js --out-dir lib
#echo -e '\003'
#node r.js -o build.js
#echo -e '\003'
#cd ..
#python main.py