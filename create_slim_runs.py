import subprocess
import os
'''
Generate replicated runs of SLiM and tabulate a binary metric from their output

Created by Nathan Oakes on 10/24/2017.
A product of the Messer Lab, http://messerlab.org/slim/

Nathan Oakes and Philipp Messer, the authors of this code, hereby place the code in this file into the public domain without restriction.  If you use this code, please credit SLiM-Extras and provide a link to the SLiM-Extras repository at https://github.com/MesserLab/SLiM-Extras.  Thank you.
'''

'''
The workflow that I find works best for me is to develop the model I like using SLiMgui, and then when I want to generate statistics, I will call SLiM inside of a python script and generate replicates in that way.  First, in my SLiM input file, inside a late() callback, I have SLiM check if a mutation of interest has gone to fixation, and then print a 1 or 0 depending on the case.  For example:

1:10000 late() {
    if (sim.countOfMutationsOfType(m1) == 0){
        fixed = (sum(sim.substitutions.mutationType == m1) == 1);
        if(fixed){
            cat('#OUTPUT: 1\n');
        } else {
            cat('#OUTPUT: 0\n');
        }
        sim.simulationFinished();
    } else {
        cat(sum(sim.mutationFrequencies(p1, sim.mutationsOfType(m1))) + "\n");
    }
}

The output string begins with "#OUTPUT:" so that when I parse the results I can search for that string.

This python script is written in python3 and requires that SLiM be in your PATH. When it is called it runs a loop that calls SLiM with the input file, produces the output, parses the output, and then adds the results to a counter when can then be used to estimate fixation probability.
'''

def parse(out):
    lines = out.split("\n")
    for line in lines:
        if line.startswith("#OUTPUT:"):
            return float(line.split()[1])

def main():
    main_dir = '/mnt/sda/home/ludeep/Desktop/PopGen/EvolutionaryGWAS/Slim_simulations/StabalizingSelectionSlim/'
    #main_dir = '/project2/jjberg/mehta5/EvolutionaryGWAS/SlimScripts/stabalizing_slim_src/'
    count = 0
    exprun='Slim_Run_Experiment_1_5_e_6'
    if not (os.path.isdir(exprun)):
        try:
            os.mkdir(exprun)
        except OSError:
            print("Error in making directory")
    for i in range(1, 500):
        slimrun = '{}/slim_run_number_{}'.format(exprun,i)
        if not (os.path.isdir(slimrun)):
            try:
                os.mkdir(slimrun)
            except OSError:
                print("Error in making directory")
        change_to = main_dir + slimrun
        os.chdir(change_to)    
        process = subprocess.Popen(["slim", "../../stabalizing_selection.slim"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,universal_newlines=True)
        out, err = process.communicate()
        #count += parse(out)
        #print(str(count/i))
        print("Slim run: {}".format(i))
        os.chdir(main_dir)

if __name__=="__main__":
    main()
