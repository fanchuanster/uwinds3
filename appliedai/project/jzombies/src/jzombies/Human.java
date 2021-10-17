/**
 * 
 */
package jzombies;

import java.awt.geom.Point2D;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import repast.simphony.engine.schedule.ScheduledMethod;
import repast.simphony.engine.watcher.Watch;
import repast.simphony.engine.watcher.WatcherTriggerSchedule;
import repast.simphony.query.space.grid.GridCell;
import repast.simphony.query.space.grid.GridCellNgh;
import repast.simphony.random.RandomHelper;
import repast.simphony.space.SpatialMath;
import repast.simphony.space.continuous.ContinuousSpace;
import repast.simphony.space.continuous.NdPoint;
import repast.simphony.space.grid.Grid;
import repast.simphony.space.grid.GridPoint;
import repast.simphony.util.SimUtilities;

/**
 * @author donwen
 *
 */
public class Human {

	protected ContinuousSpace<Object> space;
	protected Grid<Object> grid;
	protected boolean masked;
	protected int distancingSense;
	protected double activity;
	protected Vaccination vaccination;
	
	public Human(ContinuousSpace<Object> space, Grid<Object> grid, 
					boolean masked, int distancingSense, double activity, Vaccination vaccination) {
		this.space = space;
		this.grid = grid;
		this.masked = masked;
		this.distancingSense = distancingSense;
		this.activity = activity;
		this.vaccination = vaccination;
	}
	
	protected Human(Human other) {
		this.space = other.space;
		this.grid = other.grid;
		this.masked = other.masked;
		this.distancingSense = other.distancingSense;
		this.activity = other.activity;
		this.vaccination = other.vaccination;
	}
	
	public boolean getMasked() {
		return masked;
	}
	public Vaccination getVaccination() {
		return vaccination;
	}
	
	private boolean safe() {
		NdPoint thisNd = space.getLocation(this);
		GridPoint pt = grid.getLocation(this);
		
		GridCellNgh<Human> nghCreator = new GridCellNgh<Human>(grid, pt, Human.class, (int)distancingSense, (int)distancingSense);
		List<GridCell<Human>> gridCells = nghCreator.getNeighborhood(true);
		
		SimUtilities.shuffle(gridCells, RandomHelper.getUniform());
		
		boolean safe = true;
		for (GridCell<Human> cell: gridCells) {
			for (Human human: cell.items()) {
				if (human.equals(this)) {
					continue;
				}
				
				NdPoint humanNd = space.getLocation(human);
				if (space.getDistance(thisNd, humanNd) < this.distancingSense) {
					safe = false;
				}
			}
		}
		return safe;
	}
	
//	
//	@Watch(watcheeClassName = "jzombies.Zombie",
//			watcheeFieldNames = "moved",
//			query = "within_moore 1",
//			whenToTrigger = WatcherTriggerSchedule.IMMEDIATE)
	@ScheduledMethod(start = 1, interval = 1)
	public void step() {
		boolean currentlySafe = safe();
		if (currentlySafe) {
			if (RandomHelper.nextDoubleFromTo(0, 1.0) > this.activity) {
				// safe && don not want to move (according to activity)
				return;
			}
		}
		
		List<Integer> xarr = new ArrayList<Integer>();
		xarr.add(-1);
		xarr.add(0);
		xarr.add(1);
		List<Integer> yarr = new ArrayList<Integer>(xarr);
		
		SimUtilities.shuffle(xarr, RandomHelper.getUniform());
		SimUtilities.shuffle(yarr, RandomHelper.getUniform());

		GridPoint pt = grid.getLocation(this);
		NdPoint currentNd = space.getLocation(this);
		boolean isNewSafe = false;
		for (int x: xarr) {
			for (int y: yarr) {
				if (x == 0 && y == 0) {
					continue;
				}
				
				GridPoint neighborPoint = new GridPoint(pt.getX() + x, pt.getY() + y);
				moveTowards(neighborPoint);
				isNewSafe = safe();
				if (isNewSafe) {
					break;
				} else {
					// moving back
					space.moveTo(this, currentNd.getX(), currentNd.getY());
					alignGrid(this);
				}
			}
			if (isNewSafe) {
				break;
			}
		}
	}
	
	protected void alignGrid(Object object) {
		NdPoint myNewPoint = space.getLocation(object);
		grid.moveTo(object, (int)myNewPoint.getX(), (int)myNewPoint.getY());
	}
	
	public void moveTowards(GridPoint pt) {
		// only move to a point if it is not the current location.
		if (pt.equals(grid.getLocation(this))) {
			return;
		}
		
		NdPoint mySpacePoint = space.getLocation(this);
		NdPoint targetSpacePoint = new NdPoint(pt.getX(), pt.getY());
		
		double angle = SpatialMath.calcAngleFor2DMovement(space, mySpacePoint, targetSpacePoint);
		space.moveByVector(this, 1, angle, 0);
		
		alignGrid(this);
	}
}
