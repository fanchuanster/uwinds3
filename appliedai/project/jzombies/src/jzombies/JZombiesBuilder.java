/**
 * 
 */
package jzombies;

import repast.simphony.context.Context;
import repast.simphony.context.space.continuous.ContinuousSpaceFactory;
import repast.simphony.context.space.continuous.ContinuousSpaceFactoryFinder;
import repast.simphony.context.space.graph.NetworkBuilder;
import repast.simphony.context.space.grid.GridFactory;
import repast.simphony.context.space.grid.GridFactoryFinder;
import repast.simphony.dataLoader.ContextBuilder;
import repast.simphony.engine.environment.RunEnvironment;
import repast.simphony.parameter.Parameters;
import repast.simphony.random.RandomHelper;
import repast.simphony.space.continuous.ContinuousSpace;
import repast.simphony.space.continuous.NdPoint;
import repast.simphony.space.continuous.RandomCartesianAdder;

import repast.simphony.space.grid.Grid;
import repast.simphony.space.grid.GridBuilderParameters;
import repast.simphony.space.grid.SimpleGridAdder;
import repast.simphony.space.grid.WrapAroundBorders;

/**
 * @author donwen
 *
 */
public class JZombiesBuilder implements ContextBuilder<Object> {

	@Override
	public Context build(Context<Object> context) {
		context.setId("jzombies");
		
		ContinuousSpaceFactory spaceFactory =
				ContinuousSpaceFactoryFinder.createContinuousSpaceFactory (null);
		
		ContinuousSpace<Object> space =
				spaceFactory.createContinuousSpace ("space", context,
						new RandomCartesianAdder<Object>(),
						new repast.simphony.space.continuous.WrapAroundBorders(),
						50 , 50);
				
		GridFactory gridFactory = GridFactoryFinder.createGridFactory(null);
				 // Correct import : import repast.simphony.space.grid.WrapAroundBorders;
		Grid<Object> grid = gridFactory.createGrid("grid", context,
						new GridBuilderParameters<Object>(new WrapAroundBorders(),
						new SimpleGridAdder<Object>(),
						true, 50, 50));
		
		NetworkBuilder<Object> netBuilder = new NetworkBuilder<Object>("infection network", context , true);
		netBuilder.buildNetwork();


		for (int i=0; i<Settings.getHumanCount(); i++) {
			int vaccinatedDaysAgo = RandomHelper.nextIntFromTo(90, 180);
			double protectionRate = (double) RandomHelper.nextDoubleFromTo(0.9f, 0.96f);
			boolean masked = RandomHelper.nextIntFromTo(0, 100) < Settings.getMaskPercentage();
			boolean vaccinated = RandomHelper.nextIntFromTo(0, 100) < Settings.getVaccinationRate();
			context.add(new Human(space, grid, masked, Settings.getDistancingSense(), 0.5f, 
					vaccinated ? new Vaccination(vaccinatedDaysAgo, protectionRate) : null));
		}
		
		for (int i=0; i<Settings.getZombieCount(); i++) {
			Human human = (Human)context.getRandomObject();
			context.add(new Zombie(human));
		}

		int counter = 0;
		for (Object obj: context) {
			counter++;
			NdPoint pt = space.getLocation(obj);
			grid.moveTo(obj, (int)pt.getX(), (int)pt.getY());
		}
		System.out.printf("%d objects\n", counter);
		return context;
	}
}
