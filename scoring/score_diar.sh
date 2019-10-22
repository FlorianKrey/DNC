#!/bin/bash

# =======================================================
# Diarisation scoring 
# 
# For the MGB Challenge  http://www.mgb-challenge.org/
# 
# Develpped as part of the EPSRC programme in Natural 
# Speech technology 
# =======================================================
# Oscar Saz (2015) http://staffwww.dcs.shef.ac.uk/people/O.Saztorralba/

# Possible ways of input
# 1. Single STM file
# 2. Single RTTM file
# 3. Multiple RTTM files

# Possible ways of operating
# linked: Will score shows looking for links among speakers across episodes
# notlinked: Will score shows without looking for links among speakers across episodes

while [[ $# > 1 ]]
do
	key="$1"
	
	case $key in
		-r)
		reference="$2"
		shift
		;;
		-s)
		input="$2"
		shift
		;;
		-o)
		output="$2"
		shift
		;;
		-m)
		mode="$2"
		shift
		;;
		*)
		echo "Option "$key" does not exist. Usage is:"
		echo "score_diar_mgb.sh -r reference_rttm -s system_output -o output_folder -m [linked|notlinked]"
		exit
		;;
	esac
	shift
done

# Extract the list of episodes from the reference
episode_list=`awk '{print $2}' $reference | sort -u`
# Extract the list of shows from the reference
show_list=`for f in $episode_list; do echo $f | awk -F'_' '{$1=""; $2=""; $3=""; print $0}' | sed 's/^ *//g' | sed 's/ /_/g'; done | sort -u`

if [ "$mode" == "linked" ]
then
	echo "Working in \"linked\" mode..."
else
	if [ "$mode" == "notlinked" ]
	then
		echo "Working in \"notlinked\" mode..."
	else
		echo "Mode \"$mode\" is not a valid mode. Use \"linked\" or \"notlinked\""
		exit
	fi
fi

# Delete and create output folder
if [ -d $output ]
then
	echo -n "Output folder $output exists. Folder will be deleted. Continue? [y/N] "
	read text
	if [ $text == "N" ]
	then
		exit
	fi
fi
rm -rf $output
mkdir -p $output

# Verify the input, convert as required
if [ ! -e $input ]
then
	echo "Input $input does not exist either as file or folder. Terminating"
	exit
fi
if [ -f $input ]
then	
	extension=`echo $input | awk -F'.' '{print $NF}'`
	if [ $extension == "stm" -o $extension == "STM" ]
	then
		# If working with single STM file, convert to multiple RTTM files
		for episode in ${episode_list[@]}
		do
			grep $episode $input | awk -v CONVFMT=%.17g '{print "SPEAKER '$episode' 1 "$4" "$5-$4" <NA> <NA> "$3" <NA>"}' > $output/$episode.rttm
		done
	else
		if [ $extension == "rttm" -o $extension == "RTTM" ]
		then
			#If working with single RTTM file, split
			for episode in ${episode_list[@]}
			do
				grep $episode $input > $output/$episode.rttm
			done 
		else
			echo "Input file is not an STM file. Terminating"
			exit
		fi
	fi
else
	# If working with multiple RTTM files, copy to working folder
	for episode in ${episode_list[@]}
	do
		cp $input/$episode.rttm $output/$episode.rttm
	done	
fi

echo "================================================================="
echo "RESULTS FOR INDIVIDUAL SHOWS"
echo "================================================================="
composition=0
for episode in ${episode_list[@]}
do
	if [ -e $output/$episode.rttm ]
	then
		# Score each episode
		grep $episode $reference > $output/$episode.ref.rttm
		./md-eval.pl -1 -c 0.25 -r $output/$episode.ref.rttm -s $output/$episode.rttm > $output/$episode.score
		echo "Results for episode \"$episode\""
		total_time=`grep "SCORED SPEAKER TIME" $output/$episode.score | awk '{print $5}'`
		missed_time=`grep "MISSED SPEAKER TIME" $output/$episode.score | awk '{print $5}'`
		falarm_time=`grep "FALARM SPEAKER TIME" $output/$episode.score | awk '{print $5}'`
		error_time=`grep "SPEAKER ERROR TIME" $output/$episode.score | awk '{print $5}'`
		missed_time=`echo "100.00 * $missed_time / $total_time" | bc -l`
		falarm_time=`echo "100.00 * $falarm_time / $total_time" | bc -l`
		error_time=`echo "100.00 * $error_time / $total_time" | bc -l`
		der=`echo $missed_time + $falarm_time + $error_time | bc -l`
		echo -ne "TOTAL REFERENCE SPEAKER TIME: "
		printf %2.2f $total_time
		echo " seconds"
		echo -ne "MISSED SPEECH: "
	        printf %2.2f $missed_time
        	echo -ne "%\tFALSE SPEECH: "
	        printf %2.2f $falarm_time
        	echo -ne "%\tSPEAKER ERROR: "
	        printf %2.2f $error_time
        	echo "%"
	        echo -ne "DIARISATION ERROR RATE: "
        	printf %2.2f $der
	        echo "%"

		# Add the episode to the show RTTM file
		show=`echo $episode | awk -F'_' '{$1=""; $2=""; $3=""; print $0}' | sed 's/^ *//g' | sed 's/ /_/g'`
		if [ $mode == "linked" ]
		then
			#composition=`grep $episode ./scripts/composition | awk '{print $2}'`
			composition=`echo $composition + 10000 | bc -l`
			grep -h "SPKR-INFO" $output/$episode.ref.rttm | awk '{$2="'$show'"; print $0}' | sort -u >> $output/$show.linked.ref.rttm
			grep -h "SPKR-INFO" $output/$episode.rttm | awk '{$2="'$show'"; print $0}' | sort -u >> $output/$show.linked.rttm
		        grep -h "SPEAKER" $output/$episode.rttm | awk -v CONVFMT=%.17g '{$2="'$show'"; $4+='$composition'; print $0}' >> $output/$show.linked.rttm
			grep -h "SPEAKER" $output/$episode.ref.rttm | awk -v CONVFMT=%.17g '{$2="'$show'"; $4+='$composition'; print $0}' >> $output/$show.linked.ref.rttm
		else
			# Add the episode to the show RTTM file (not considering speaker linking)
			cat $output/$episode.rttm >> $output/$show.notlinked.rttm
			cat $output/$episode.ref.rttm >> $output/$show.notlinked.ref.rttm
			# Add the episode to the global RTTM file (not considering speaker linking)
                        cat $output/$episode.rttm >> $output/complete.notlinked.rttm
			cat $output/$episode.ref.rttm >> $output/complete.notlinked.ref.rttm
		fi
	else
		# There is no system output for a given episode. Warning!
		echo "System output does not exist for episode $episode"
		echo "Scoring will continue, but this episode will be scored as missed speaker"
	fi
done

echo "================================================================="
echo "RESULTS FOR SERIES"
echo "================================================================="
for show in ${show_list[@]}
do
	# Score each show
	grep -h "SPKR-INFO" $output/$show.$mode.rttm | sort -u > $output/temp
	sed '/SPKR-INFO/d' $output/$show.$mode.rttm > $output/temp2
	cat $output/temp $output/temp2 > $output/$show.$mode.rttm
	grep -h "SPKR-INFO" $output/$show.$mode.ref.rttm | sort -u > $output/temp
        sed '/SPKR-INFO/d' $output/$show.$mode.ref.rttm > $output/temp2
        cat $output/temp $output/temp2 > $output/$show.$mode.ref.rttm
	./md-eval.pl -1 -c 0.25 -r $output/$show.$mode.ref.rttm -s $output/$show.$mode.rttm > $output/$show.$mode.score

	echo "Results for show \"$show\""
        total_time=`grep "SCORED SPEAKER TIME" $output/$show.$mode.score | awk '{print $5}'`
        missed_time=`grep "MISSED SPEAKER TIME" $output/$show.$mode.score | awk '{print $5}'`
        falarm_time=`grep "FALARM SPEAKER TIME" $output/$show.$mode.score | awk '{print $5}'`
        error_time=`grep "SPEAKER ERROR TIME" $output/$show.$mode.score | awk '{print $5}'`
        missed_time=`echo "100.00 * $missed_time / $total_time" | bc -l`
        falarm_time=`echo "100.00 * $falarm_time / $total_time" | bc -l`
        error_time=`echo "100.00 * $error_time / $total_time" | bc -l`
        der=`echo $missed_time + $falarm_time + $error_time | bc -l`
	echo -ne "TOTAL REFERENCE SPEAKER TIME: "
        printf %2.2f $total_time
        echo " seconds"
	echo -ne "MISSED SPEECH: "
	printf %2.2f $missed_time
	echo -ne "%\tFALSE SPEECH: "
	printf %2.2f $falarm_time
	echo -ne "%\tSPEAKER ERROR: "
	printf %2.2f $error_time
	echo "%"
	echo -ne "DIARISATION ERROR RATE: "
	printf %2.2f $der
	echo "%"

	if [ $mode == "linked" ]
	then
		# Add the show to the global RTTM file (considering speaker linking)
		cat $output/$show.linked.rttm >> $output/complete.linked.rttm
		cat $output/$show.linked.ref.rttm >> $output/complete.linked.ref.rttm
	fi
done

# Global scoring
./md-eval.pl -1 -c 0.25 -r $output/complete.$mode.ref.rttm -s $output/complete.$mode.rttm > $output/complete.$mode.score

echo "================================================================="
echo "GLOBAL RESULTS"
echo "================================================================="
total_time=`grep "SCORED SPEAKER TIME" $output/complete.$mode.score | awk '{print $5}'`
missed_time=`grep "MISSED SPEAKER TIME" $output/complete.$mode.score | awk '{print $5}'`
falarm_time=`grep "FALARM SPEAKER TIME" $output/complete.$mode.score | awk '{print $5}'`
error_time=`grep "SPEAKER ERROR TIME" $output/complete.$mode.score | awk '{print $5}'`
missed_time=`echo "100.00 * $missed_time / $total_time" | bc -l`
falarm_time=`echo "100.00 * $falarm_time / $total_time" | bc -l`
error_time=`echo "100.00 * $error_time / $total_time" | bc -l`
der=`echo $missed_time + $falarm_time + $error_time | bc -l`
echo -ne "TOTAL REFERENCE SPEAKER TIME: "
printf %2.2f $total_time
echo " seconds"
echo -ne "MISSED SPEECH: "
printf %2.2f $missed_time
echo -ne "%\tFALSE SPEECH: "
printf %2.2f $falarm_time
echo -ne "%\tSPEAKER ERROR: "
printf %2.2f $error_time
echo "%"
echo -ne "DIARISATION ERROR RATE: "
printf %2.2f $der
echo "%"
echo "================================================================="

#Clean afterwards
rm -rf $output/temp $output/temp2
rm -rf $output/*.rttm

