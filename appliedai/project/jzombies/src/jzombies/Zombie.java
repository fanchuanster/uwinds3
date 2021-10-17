/**
 * 
 */
package jzombies;

import java.util.ArrayList;
import java.util.List;

import repast.simphony.engine.schedule.ScheduledMethod;
import repast.simphony.query.space.grid.GridCell;
import repast.simphony.query.space.grid.GridCellNgh;
import repast.simphony.random.RandomHelper;
import repast.simphony.space.SpatialMath;
import repast.simphony.space.continuous.ContinuousSpace;
import repast.simphony.space.continuous.NdPoint;
import repast.simphony.space.graph.Network;
import repast.simphony.space.grid.Grid;
import repast.simphony.space.grid.GridPoint;
import repast.simphony.util.ContextUtils;
import repast.simphony.util.SimUtilities;
import repast.simphony.context.Context;

/**
 * @author donwen
 *
 */
public class Zombie extends Human {
	private int incubationTicks;
	
	public Zombie(Human human) {
		super(human);
		incubationTicks = Settings.getIncubationTicks();
	}
	
	@ScheduledMethod(start = 1, interval = 1)
	@Override
	public void step() {
		if (this.incubationTicks-- <= 0) {
			// once symptomatic, doing nothing.
			return;
		}
		
		super.step();
		
		infect();
	}	

	public void infect() {
		GridPoint pt = grid.getLocation(this);
		List<Human> humans = new ArrayList<Human>();
		
		GridCellNgh<Human> nghCreator = new GridCellNgh<Human>(grid, pt, Human.class, 4, 4);
		List<GridCell<Human>> gridCells = nghCreator.getNeighborhood(true);
		
		for (GridCell<Human> cell: gridCells) {
			cell.items().forEach(item -> {
				humans.add(item);
			});
		}

		for (Human human: humans) {
			if (human instanceof Zombie) {
				continue;
			}
			
			boolean masked1 = this.getMasked();
			boolean masked2 = human.getMasked();
			NdPoint spacePt1 = space.getLocation(this);
			NdPoint spacePt2 = space.getLocation(human);
			
			assert(human != null);
			assert(this != null);
			assert(spacePt1 != null);
			assert(spacePt2 != null);
			if (spacePt1 == null) {
				System.out.println("null");
			}
			
//			System.out.println(spacePt1.dimensionCount());
//			System.out.println(spacePt2.dimensionCount());
			
			double distance = space.getDistance(spacePt1, spacePt2);
			double infectionRate = Infection.getInfectionRate(distance, masked1, masked2, human.getVaccination());
			double randomRate = RandomHelper.nextDoubleFromTo(0, 1.0);
			boolean infected = randomRate <= infectionRate;
			
			if (infected) {
				System.out.printf("infected one with randomRate %f <= infectionRate %f\n", randomRate, infectionRate);
				NdPoint spacePt = space.getLocation(human);
				
				Context<Object> context = ContextUtils.getContext(human);
				context.remove(human);
				
				Zombie newZombie = new Zombie(human);
				context.add(newZombie);
				space.moveTo(newZombie, spacePt.getX(), spacePt.getY());
				alignGrid(newZombie);
				
				Network<Object> net = (Network<Object>)context.getProjection("infection network");
				net.addEdge(this, newZombie);				
			}
		}
	}
}
