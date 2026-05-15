from src.simulation.traffic_simulator import ProductionTrafficSimulator


MAX_ROWS = 25000
# 6000 + 25000

def main() -> None:
    simulator = ProductionTrafficSimulator()
    simulator.run(max_rows=MAX_ROWS, start_row=6000)


if __name__ == "__main__":
    main()
