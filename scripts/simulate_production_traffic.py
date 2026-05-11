from src.simulation.traffic_simulator import ProductionTrafficSimulator


MAX_ROWS = 1000


def main() -> None:
    simulator = ProductionTrafficSimulator()
    simulator.run(max_rows=MAX_ROWS)


if __name__ == "__main__":
    main()
