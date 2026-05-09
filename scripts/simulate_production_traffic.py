from src.simulation.traffic_simulator import ProductionTrafficSimulator

def main() -> None:
    simulator = ProductionTrafficSimulator()
    simulator.run(max_rows = 1000)

if __name__ == '__main__':
    main()