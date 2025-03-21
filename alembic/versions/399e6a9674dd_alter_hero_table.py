"""Alter Hero Table

Revision ID: 399e6a9674dd
Revises: 4769139e995d
Create Date: 2025-03-12 16:48:49.937698

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel


# revision identifiers, used by Alembic.
revision: str = '399e6a9674dd'
down_revision: Union[str, None] = '4769139e995d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('hero', sa.Column('test1', sqlmodel.sql.sqltypes.AutoString(), nullable=True))
    # ### end Alembic commands ###


def downgrade() -> None:
    """Downgrade schema."""
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column('hero', 'test1')
    # ### end Alembic commands ###
