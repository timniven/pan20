import sqlalchemy as db
from sqlalchemy import orm
from sqlalchemy.ext import declarative


Base = declarative.declarative_base()


doc_ngram = db.Table(
    'doc_ngram',
    Base.metadata,
    db.Column('doc_id', db.String, db.ForeignKey('doc.id')),
    db.Column('gram', db.String, db.ForeignKey('ngram.gram'))
)


class Doc(Base):
    __tablename__ = 'doc'
    id = db.Column(db.String, primary_key=True)
    text = db.Column(db.String, nullable=False)
    fandom_name = db.Column(
        db.String, db.ForeignKey('fandom.name'), nullable=False)
    fandom = orm.relationship(
        'Fandom',
        backref='docs',
        foreign_keys=[fandom_name])
    author_id = db.Column(
        db.Integer, db.ForeignKey('author.id'), nullable=False)
    author = orm.relationship(
        'Author',
        backref='authors',
        foreign_keys=[author_id])
    ngrams = orm.relationship('NGram', secondary=doc_ngram)


class Author(Base):
    __tablename__ = 'author'
    id = db.Column(db.String, primary_key=True)


class Fandom(Base):
    __tablename__ = 'fandom'
    name = db.Column(db.String, primary_key=True)


class NGram(Base):
    __tablename__ = 'ngram'
    gram = db.Column(db.String, primary_key=True)
    order = db.Column(db.Integer, nullable=False)
    docs = orm.relationship('Doc', secondary=doc_ngram)
