#Neuro-evolution of augmenting topologies in evolving return launch vehicles in KSP, version 0.1.7.

##>>>>> REQUIREMENTS <<<<
##  - Python 3.6.1
##  - Kerbal Space Program 1.2.2
##  - kRPC 0.3.7
##  - neat-python 0.91

##TODO: - Write and insert Euler's method for trajectory prediction for upper atmosphere return behaviour.
##      - Improve visualization methods
##      - Expand on save states (individual organisms, species history)

import math
import os
import time
import krpc
import neat
import visualize


conn = krpc.connect(name='NEAT RLV Recovery')

# Set vessel and target properties.
# Launch platform   height = 6      lat: -0.09722817157371434 lon: -74.55767181444384
# VAB               height = 103    lat: -0.09626606722596398 lon: -74.61882882582802
# Vessel altitude on the altitude meter is 10 m higher than actual surface altitude
vessel_height = 75
partlist = 47
target_height = 6
target_latitude = -0.09722817157371434
target_longitude = -74.55767181444384

def eval_fitness(genomes, config):
    eval_fitness.organism = 0
    
    for genome_id, genome in genomes:

        net = neat.nn.FeedForwardNetwork.create(genome, config)
        eval_fitness.organism +=1

        # Load quicksave, could change to random load.
        conn.space_center.load('quicksave')
        vessel = conn.space_center.active_vessel
        ref_frame = vessel.orbit.body.reference_frame
        time.sleep(0.2)
        vessel.control.set_action_group(1, True)

        # Set vessel init parameters.
        partlist = len(vessel.parts.all)
        position = [0, 0]
        fitness_penalty = 1
        fitness_multiplier = 1
        fitness_fuel = 1
        time_landed = 0
        last_speed = 0
        second_last_speed = 0
        initial_landing = False
        
        
        while True:
            # Get vessel data.
            latitude = vessel.flight().latitude
            longitude = vessel.flight().longitude
            mean_altitude = vessel.flight().surface_altitude
            speed = vessel.flight(ref_frame).speed
            horizontal_speed = vessel.flight(ref_frame).horizontal_speed
            vertical_speed = vessel.flight(ref_frame).vertical_speed
            pitch = vessel.flight().pitch
            heading = vessel.flight().heading
            fuel = vessel.resources.amount('Oxidizer')
            altitude = mean_altitude - vessel_height - target_height
            position = [latitude, longitude]
            target_distance = get_distance(position)*5000
            target_heading = get_heading(position)
            landed = is_landed(vessel)
                
            # Check for negative end conditions
            if vertical_speed > 50:
                fitness_penalty = 20000
                print('Penalty of %s for excessive vertical speed.' % fitness_penalty)
                break
            if fuel <=0 and altitude > 10:
                fitness_penalty = altitude/10 + abs(vertical_speed)
                print('Penalty of %s for running out of fuel.' % fitness_penalty)
                break
            if len(vessel.parts.all) < partlist:
                if abs(second_last_speed) > 20:
                    fitness_penalty = abs(second_last_speed)**2.1
                    print('Penalty of %s for landing at excessive speed.' % fitness_penalty)
                    break
                elif abs(second_last_speed) > 15:
                    fitness_penalty = abs(second_last_speed)
                    print('Penalty of %s for landing at high speed.' % fitness_penalty)
                else:
                    print('Structural damage... Calculating fitness.')
                    break
            if len(vessel.parts.all) <= 0:
                print('Structural disintegration... Calculating fitness.')
                break
                
            # Check for positive end conditions.
            if landed:
                if second_last_speed <= 4 and initial_landing == False:
                    fitness_multiplier = 2.5
                elif second_last_speed <= 10 and initial_landing == False:
                    fitness_multiplier = 10/last_speed
                initial_landing = True
                vessel.control.rcs = True
                last_pitch = vessel.flight().pitch
                time.sleep(0.1)
                if abs(last_pitch - vessel.flight().pitch) >= 0.5:
                      time_landed = 0
                else:
                    time_landed += 0.1
                if time_landed >= 2:
                    fitness_multiplier *= 100
                    if fuel > 100:
                        fitness_fuel = fuel/100
                    print('Landed... Calculating fitness.')
                    break
                
            second_last_speed = last_speed
            last_speed = speed
            if altitude < 400 and vessel.control.gear == False:
                vessel.control.gear = True

            # Activate neural net.
            inputs = [target_distance, altitude, horizontal_speed, vertical_speed, pitch, landed]
            outputs = net.activate(inputs)
            process_output(vessel, position, target_heading, outputs)
    
        genome.fitness = calc_fitness(vessel, position, fitness_multiplier,fitness_fuel)/fitness_penalty
        print('Organism: ', eval_fitness.organism, '\tFitness: ', genome.fitness)

def get_heading(position):
    heading = math.degrees(math.atan2(target_longitude - position[1], target_latitude - position[0]))
    return heading

def get_distance(position):
    posx = target_latitude - position[0]
    posy = target_longitude - position[1]
    distance = math.sqrt(posx**2 + posy**2 + 0.0001)
    return distance

def is_landed(vessel):
    if vessel.situation == vessel.situation.landed:
        return True
    else:
        return False
    
def process_output(vessel, position, set_heading, output):
    ref_frame = vessel.orbit.body.reference_frame
    vessel.control.throttle = output[0]
    set_pitch = abs(output[1]*90)
    if output[2] >=0.5 and output[3] >=0.5:
        vessel.auto_pilot.engage()
        vessel.auto_pilot.target_pitch_and_heading(set_pitch, set_heading)
    if output[2] >=0.5 and output[3] <0.5 and vessel.control.sas_mode != conn.space_center.SASMode.retrograde and vessel.flight(ref_frame).speed > 2 and len(vessel.parts.all) == partlist:
        vessel.auto_pilot.disengage()
        vessel.control.sas = True
        time.sleep(0.1)
        vessel.control.sas_mode = conn.space_center.SASMode.retrograde
    if output[2] <0.5 and output[3] >=0.5 and vessel.control.sas_mode != conn.space_center.SASMode.radial:
        vessel.auto_pilot.disengage()
        vessel.control.sas = True
        time.sleep(0.1)
        vessel.control.sas_mode = conn.space_center.SASMode.radial
    if output[2] <0.5 and output[3] <0.5 and vessel.control.sas_mode != conn.space_center.SASMode.stability_assist:
        vessel.auto_pilot.disengage()
        vessel.control.sas = True
        time.sleep(0.1)
        vessel.control.sas_mode = conn.space_center.SASMode.stability_assist
        
    if output[4] >= 0.5 and vessel.control.get_action_group(5) == False:
        vessel.control.set_action_group(5, True)
    if output[4] < 0.5 and vessel.control.get_action_group(5) == True:
        vessel.control.set_action_group(5, False)
        

def calc_fitness(vessel, position, fitness_multiplier, fuel_fitness):
    # Check if the vessel is intact and facing up.
    if partlist <= 0:
        part_ratio = 0.01
        pitch_fitness = 1
    else:
        part_ratio = len(vessel.parts.all)/partlist
        if vessel.flight().pitch > 0:
            pitch_fitness = vessel.flight().pitch/90
        else:
            pitch_fitness = 0.01
    part_fitness = part_ratio**3

    # Calculate the distance to the target site.
    target_distance = get_distance(position)
    target_fitness = 1/target_distance*pitch_fitness

    landing_fitness = target_fitness*part_fitness*fitness_multiplier*fuel_fitness*pitch_fitness
    total_fitness = target_fitness + landing_fitness
    return total_fitness

def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_file)
    
    # Load or create a population.
    local_dir = os.path.dirname(__file__)
    last_checkpoint = ''
    for file in os.listdir(local_dir):
        if 'neat-checkpoint' in file:
            last_checkpoint = file
    try:
        pop = neat.Checkpointer.restore_checkpoint(last_checkpoint)
        print('Loading previous save state.')
    except:
        print('No previous save state found.')
        pop = neat.Population(config)
    # Add reporter modules.
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    # Create a checkpoint every generation.
    pop.add_reporter(neat.Checkpointer(1))

    # Run for 100 generations
    winner = pop.run(eval_fitness, 100)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Display output of the most fit genome against overall population fitness (install and import visualize module).
    #visualize.draw_net(config, winner, True)
    #visualize.plot_stats(stats, ylog=False, view=True)
    #visualize.plot_species(stats, view=True)
    
    print('Finished run')

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config')
    run(config_path)


