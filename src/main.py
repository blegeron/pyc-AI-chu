import click
from loguru import logger


@click.command()
@click.option(
    "--name",
    help="The person to greet.",
)
def main(name: any) -> None:
    """Say hello to NAME."""
    click.echo(f"Hello {name}!")
    logger.info("✅ Hello executed.")


if __name__ == "__main__":
    main()
