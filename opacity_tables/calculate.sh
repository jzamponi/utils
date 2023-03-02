#for a in 10 100 1000
#do
#    for material in "s" "sg"
#    do
#        synthesizer \
#            --opacity \
#            --nang 181 \
#            --na 200 \
#            --overwrite \
#            --amax ${a} \
#            --material ${material} \
#            --nopb
#
#        synthesizer \
#            --opacity \
#            --nang 181 \
#            --na 200 \
#            --overwrite \
#            --amax ${a} \
#            --material ${material} \
#            --polarization \
#            --nopb
#    done
#done

# Cases with organics and sublimation
for a in 10 100 1000
do
    for org in 10 30 50 80
    do
        synthesizer \
            --opacity \
            --nang 181 \
            --na 200 \
            --overwrite \
            --amax ${a} \
            --material sgo \
            --sublimation ${org} \
            --nopb

        synthesizer \
            --opacity \
            --nang 181 \
            --na 200 \
            --overwrite \
            --amax ${a} \
            --material sgo \
            --polarization \
            --sublimation ${org} \
            --nopb
    done
done
