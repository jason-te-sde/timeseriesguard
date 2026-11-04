from tsguard.cli import main


def test_main_returns_the_process_exit_code_zero():
    assert main() == 0
