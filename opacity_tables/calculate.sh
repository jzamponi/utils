for a in 10 100 1000
do
#    for material in "s" "sg" "sgo"
    for material in "sgo"
    do
        for csubl in "30" "50" "80"
        do
            synthesizer \
                --opacity \
                --nang 181 \
                --na 200 \
                --overwrite \
                --amax ${a} \
                --material ${material} \
                --sublimation ${csubl} 

            synthesizer \
                --polarization \
                --opacity \
                --nang 181 \
                --na 200 \
                --overwrite \
                --amax ${a} \
                --material ${material} \
                --sublimation ${csubl} 
        done            
    done
done
